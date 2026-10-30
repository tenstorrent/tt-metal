# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Long-prompt prefill demo on the full ttnn ``DeepSeekV41PrefillModel`` (the prefill twin of
``tests/decode/test_full_model_decode_demo.py``, modeled on V4's ``test_full_model_prefill_demo.py``).

Opens the full 8 x 4 Galaxy mesh and carves ``DEEPSEEK_V4_NUM_STAGES`` (default eight) ``1 x 4`` tensor-parallel
submeshes out of it (rows 0..N-1 of column 0). The real 40-layer V4.1 checkpoint is built on those stages (five
layers a stage at the default; the embedding with the first stage, the head with the last) and tokenizes a *real*
prompt -- by default the book prompt of
``models/tt_transformers/demo/sample_prompts/input_data_long_128k.json``, cut or repeated to
``DEEPSEEK_V4_PREFILL_LEN`` tokens -- and prefills it chunk by chunk.

By default the prefill is *KV-only*: layers after the last kv source (layer 20) only read its compressed KV, so layers
0-20 fill all of the shared compressed KV and index keys and the remaining 19 layers are not built (no head either, so
no first token). The sliding-window rings of layers 21-39 are not filled; decode would replay the last 128 tokens for
them. ``DEEPSEEK_V4_PREFILL_KV_ONLY=0`` builds the whole stack and reports the first token.

The prefill is *traced and pipelined* (:class:`~models.experimental.deepseek_v41_flash.tt.prefill.model.DeepSeekV41TracedPrefill`,
V4's ``TracedPrefill``): the chunk step is compiled and captured once per stage (not timed), then replayed for every
chunk with the stages running as a pipeline. A chunk's only host input is its packet (H2D on stage 0) and its Engram
rows (H2D on the Engram layers' stages); the residual and any compressed KV a later stage reads go stage to stage
over D2D sockets, and the last stage sends its output to the host over a D2H socket. The layers are split evenly into
contiguous runs (V4's ``plan_layer_placement`` with a pipeline group size of 1): KV-only puts its 21 layers on the
default eight stages of 3 or 2 layers (0-2, 3-5, 6-8, 9-11, 12-14, 15-16, 17-18, 19-20).

The final collapse uses the last layer's FFN ``pre_mix`` (there is no ``hc_head``). Engram layers 1 and 14 read host
n-gram tables (~101 GB each) when built, and the test is skipped when that RAM is not free.

Prompt length: 16384 tokens by default. Longer prompts run too: past ``candidate_topk_blocks * candidate_block_size``
= 16384 compressed entries the two-level select bounds what layers 20 and later score, but the index sources before
it score every entry a chunk can see, so each chunk costs more the further in it is.

Run it (ttnn venv)::

    DEEPSEEK_V4_CACHE_DIR=/path/to/cache pytest -s \\
      models/experimental/deepseek_v41_flash/tests/prefill/test_full_model_prefill_demo.py

Knobs (environment):

* ``DEEPSEEK_V4_PREFILL_LEN``     -- prompt length in tokens, a multiple of 128 (default 16384),
* ``DEEPSEEK_V4_PREFILL_CHUNK``   -- chunk size in tokens, a multiple of 128 (default 1024),
* ``DEEPSEEK_V4_PREFILL_KV_ONLY`` -- 1 (default): layers 0-20 only, no head; 0: the full stack with the head,
* ``DEEPSEEK_V4_PREFILL_LAYERS``  -- build only the first N layers (default 21 when KV-only, else all 40; with KV-only
  it must be at least 21, so use ``DEEPSEEK_V4_PREFILL_KV_ONLY=0`` for a shorter bring-up),
* ``DEEPSEEK_V4_NUM_STAGES``      -- how many ``1 x 4`` submeshes to carve from the full mesh (default 8; fewer
  are used when there are fewer layers). ``DEEPSEEK_V4_PREFILL_STAGES`` still applies when this is unset,
* ``DEEPSEEK_V4_TRACE_REGION_SIZE`` -- bytes to reserve for the captured traces (unset keeps the ttnn default),
* ``DEEPSEEK_V4_PREFILL_HEARTBEAT`` -- seconds between ``[heartbeat]`` lines saying what it is doing (default 30),
* ``DEEPSEEK_V4_PREFILL_STALL_SECS`` -- warn when one step lasts longer than this (default 600),
* ``DEEPSEEK_V4_PREFILL_PROMPT``  -- another prompt file (a JSON list; the first entry is used),
* ``DEEPSEEK_V4_CACHE_DIR``       -- the converted-weight tile cache (shared with the decode demo's routed experts).

A quick bring-up: ``DEEPSEEK_V4_PREFILL_KV_ONLY=0 DEEPSEEK_V4_PREFILL_LAYERS=3 DEEPSEEK_V4_PREFILL_LEN=1024`` (layers 0-2: a window layer, an Engram
layer and a ratio-2 kv + index source, one per stage on three stages).
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
    _ENGRAM_RAM_MARGIN,
    _available_bytes,
    _build_rope,
    _checkpoint_available,
)
from models.experimental.deepseek_v41_flash.tt.config import (
    CACHE_DIR,
    DEFAULT_MODEL_DIR,
    load_config,
    pipeline_stage_count,
)
from models.experimental.deepseek_v41_flash.tt.engram_lookup import (
    EngramHostLookup,
    engram_ram_needed,
    load_engram_table,
)
from models.experimental.deepseek_v41_flash.tt.prefill.model import DeepSeekV41PrefillModel
from models.experimental.deepseek_v4_flash.tests.prefill.test_full_model_prefill_demo import (
    _DEFAULT_PROMPT_FILE,
    _Progress,
    _build_prompt_ids,
    _env_int,
    _report,
)
from models.experimental.deepseek_v4_flash.tt.model import plan_layer_placement
from models.experimental.deepseek_v4_flash.tt.prefill.attention import ALIGNMENT
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config, set_active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

_TP_SIZE = 4
_NUM_STAGES = pipeline_stage_count()
_ATTENTION_WEIGHT_DTYPE = ttnn.bfloat8_b
_EXPERT_DTYPE = ttnn.bfloat4_b


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(21600)  # weight conversion on a cold cache + a long prefill
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            **(
                {"trace_region_size": int(os.environ["DEEPSEEK_V4_TRACE_REGION_SIZE"])}
                if os.environ.get("DEEPSEEK_V4_TRACE_REGION_SIZE")
                else {}
            ),
        }
    ],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("mesh_device", [(8, _TP_SIZE)], indirect=["mesh_device"], ids=["galaxy_8x4"])
def test_full_model_prefill_demo(mesh_device, reset_seeds) -> None:
    from transformers import AutoTokenizer

    progress = _Progress(
        interval=float(os.environ.get("DEEPSEEK_V4_PREFILL_HEARTBEAT", 30)),
        stall=float(os.environ.get("DEEPSEEK_V4_PREFILL_STALL_SECS", 600)),
    )
    with progress:
        _run_demo(mesh_device, progress, AutoTokenizer)


def _run_demo(mesh_device, progress: _Progress, AutoTokenizer) -> None:
    target_len = _env_int("DEEPSEEK_V4_PREFILL_LEN", 16384)
    chunk_size = _env_int("DEEPSEEK_V4_PREFILL_CHUNK", 1024)
    for name, value in (("DEEPSEEK_V4_PREFILL_LEN", target_len), ("DEEPSEEK_V4_PREFILL_CHUNK", chunk_size)):
        if value <= 0 or value % ALIGNMENT:
            raise ValueError(f"{name}={value} must be a positive multiple of {ALIGNMENT}")
    logger.info(
        f"settings: prompt {target_len} tokens, chunk {chunk_size}, traced, heartbeat every {progress.interval:.0f}s, "
        f"stall warning after {progress.stall:.0f}s, per-layer messages for the build"
    )

    progress.step("[1/6] reading the checkpoint index, the config and the tokenizer")
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    kv_only = bool(_env_int("DEEPSEEK_V4_PREFILL_KV_ONLY", 1))
    default_layers = max(config.kv_source_layer_ids) + 1 if kv_only else config.num_hidden_layers
    num_layers = min(_env_int("DEEPSEEK_V4_PREFILL_LAYERS", default_layers), config.num_hidden_layers)
    logger.info(
        f"checkpoint {loader.snapshot_dir}: {num_layers}/{config.num_hidden_layers} layers will be built{' (KV-only: no head)' if kv_only else ''}"
    )

    engram_ids = [i for i in config.engram_layer_ids if i < num_layers]
    if engram_ids:
        need = _ENGRAM_RAM_MARGIN * engram_ram_needed(loader, engram_ids)
        available = _available_bytes()
        if available < need:
            pytest.skip(
                f"Engram tables for layers {engram_ids} need {need / 1e9:.0f} GB with headroom; "
                f"{available / 1e9:.0f} GB is available"
            )

    # --- the prompt ------------------------------------------------------------------------------ #
    progress.step(f"[2/6] building the {target_len}-token prompt (downloads the book on first use, tokenizes it)")
    prompt_file = Path(os.environ.get("DEEPSEEK_V4_PREFILL_PROMPT", _DEFAULT_PROMPT_FILE))
    entry = json.loads(prompt_file.read_text())[0]
    prompt_ids, info = _build_prompt_ids(tokenizer, entry, target_len)
    assert len(prompt_ids) % ALIGNMENT == 0, f"prompt of {len(prompt_ids)} tokens is not a multiple of {ALIGNMENT}"
    repeated = f", its first {info['repeated']} repeated to fill" if info["repeated"] else ""
    logger.info(
        f"prompt file {prompt_file.name}: {len(prompt_ids)} tokens = {info['prefix']} template/fence + "
        f"{info['context']} context ({info['book']} in the book{repeated}) + {info['suffix']} instruction"
    )
    logger.info(f"prompt starts : {tokenizer.decode(prompt_ids[:24])!r}")
    logger.info(f"prompt ends   : {tokenizer.decode(prompt_ids[-90:])!r}")
    num_chunks = math.ceil(len(prompt_ids) / chunk_size)
    logger.info(f"prefill plan: {num_chunks} chunks of {chunk_size} tokens")

    # --- the mesh: the full Galaxy is open; each pipeline stage is a 1 x TP submesh of it -------- #
    progress.step("[3/6] system profile and submeshes")
    system_config = load_system_config(mesh_device=mesh_device).log()
    set_active_system_config(system_config)
    mesh_rows, mesh_cols = tuple(mesh_device.shape)
    assert mesh_cols >= _TP_SIZE, f"a 1 x {_TP_SIZE} submesh does not fit in the {mesh_rows} x {mesh_cols} mesh"
    num_stages = _NUM_STAGES
    if not 1 <= num_stages <= mesh_rows:
        raise ValueError(f"DEEPSEEK_V4_NUM_STAGES={num_stages} must be in [1, {mesh_rows}] (mesh rows)")
    placement = plan_layer_placement(num_layers, num_stages, 1)
    num_stages = placement[-1] + 1
    logger.info(f"opened a {mesh_rows} x {mesh_cols} mesh; {num_stages} stage(s) of 1 x {_TP_SIZE}")
    submeshes = []
    for i in range(num_stages):
        submeshes.append(mesh_device.create_submesh(ttnn.MeshShape(1, _TP_SIZE), ttnn.MeshCoordinate(i, 0)))
        logger.info(f"created submesh for stage {i}: 1 x {_TP_SIZE} at mesh row {i}, column 0")
    layer_devices = [submeshes[k] for k in placement]
    spans = []
    for k in dict.fromkeys(placement):
        owned = [li for li, stage in enumerate(placement) if stage == k]
        spans.append(f"stage {k}: layers {owned[0]}-{owned[-1]}")
    logger.info(
        f"{num_layers}/{config.num_hidden_layers} layers over {num_stages} stages x TP{_TP_SIZE}: " + ", ".join(spans)
    )

    progress.step(f"[4/6] RoPE tables for {len(prompt_ids)} positions")
    t0 = time.perf_counter()
    rope = _build_rope(config, loader.snapshot_dir, len(prompt_ids))
    logger.info(f"RoPE tables built in {time.perf_counter() - t0:.1f}s")

    engram_lookup = None
    if engram_ids:
        progress.step(f"[5/6] Engram tables of layers {engram_ids} -> host RAM (~101 GB each)")
        t0 = time.perf_counter()
        tables = {i: load_engram_table(loader, i) for i in engram_ids}
        engram_lookup = EngramHostLookup(
            config, loader.snapshot_dir, tables, batch=1, max_seq=len(prompt_ids), tokenizer=tokenizer
        )
        logger.info(f"Engram tables loaded in {time.perf_counter() - t0:.1f}s")

    # --- the model ------------------------------------------------------------------------------- #
    progress.step(
        f"[6/6] building the model: {num_layers} layers (a cold weight cache reads + quantizes the checkpoint: "
        "slow; a warm one just uploads), then the prefill"
    )
    cache = WeightCache(CACHE_DIR) if CACHE_DIR else None
    logger.info(f"weight cache: {cache.path if cache else 'disabled'}")
    t0 = time.perf_counter()
    model = DeepSeekV41PrefillModel(
        config,
        loader,
        layer_devices[0],
        rope,
        num_layers=num_layers,
        cache=cache,
        weight_dtype=_ATTENTION_WEIGHT_DTYPE,
        expert_dtype=_EXPERT_DTYPE,
        tp_size=_TP_SIZE,
        layer_devices=layer_devices,
        progress=progress,
        engram_lookup=engram_lookup,
        kv_only=kv_only,
    )
    progress.step("model built; waiting for every device to finish its uploads")
    model.synchronize("uploads")
    build_seconds = time.perf_counter() - t0
    logger.info(f"built DeepSeekV41PrefillModel ({model.num_layers} layers) in {build_seconds:.1f}s")

    ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)

    # --- traces: persistent state and sockets, compile the chunk step, capture (not timed) -------- #
    # Before anything else allocates on the devices: allocating on a device that holds a trace is unsafe.
    progress.step(
        f"preparing the traced prefill for {len(prompt_ids)} tokens in chunks of {chunk_size}: persistent buffers, "
        "H2D / D2D / D2H sockets, compile the chunk step, capture one trace per stage (not timed)"
    )
    t0 = time.perf_counter()
    model.prepare_traced_prefill(len(prompt_ids), chunk_size)
    prepare_seconds = time.perf_counter() - t0
    logger.info(f"traced prefill prepared in {prepare_seconds:.1f}s")

    # --- the measured prefill: trace replays, the stages pipelined -------------------------------- #
    progress.step(
        f"prefill: {num_chunks} chunks of {chunk_size} tokens through {num_stages} pipelined stage(s) (timed)"
    )
    chunk_times: list[tuple[int, int, float]] = []
    elapsed = 0.0

    def on_chunk(index: int, start: int, end: int, seconds: float) -> None:
        nonlocal elapsed
        elapsed += seconds
        chunk_times.append((start, end, seconds))
        recent = [s for _, _, s in chunk_times[1:][-8:]] or [seconds]
        remaining = (num_chunks - index - 1) * statistics.mean(recent)
        fill = " (pipeline fill)" if index == 0 and num_chunks > 1 else ""
        logger.info(
            f"prefill chunk {index + 1}/{num_chunks} tokens [{start}, {end}) out: +{seconds:.3f} s{fill} | "
            f"{seconds / (end - start) * 1000:.3f} ms/token | {(end - start) / seconds:.1f} tok/s | "
            f"{end}/{len(prompt_ids)} tokens, elapsed {elapsed:.1f} s, ~{remaining:.0f} s to go"
        )

    # The replay runs no per-layer Python, so only the chunk lines (and the heartbeat) remain.
    progress.verbose = False
    t0 = time.perf_counter()
    logits, states = model.prefill_traced(ids, on_chunk=on_chunk)
    wall_seconds = time.perf_counter() - t0
    progress.step("prefill finished")
    _report(chunk_times, num_stages, wall_seconds, build_seconds, prepare_seconds)

    assert len(chunk_times) == num_chunks
    assert all(state.seq_len == len(prompt_ids) for state in states)

    if kv_only:
        # Every kv source must hold its compressed entries (one per ``rate`` tokens) and index keys, all finite.
        assert logits is None
        for i in config.kv_source_layer_ids:
            state, rate = states[i], config.compress_ratios[i]
            assert state.num_entries == len(prompt_ids) // rate, (i, state.num_entries, rate)
            for name in ("compressed_kv", "idx_keys"):
                host = model.to_host(getattr(state, name), model.layer_devices[i])
                assert host.shape[2] == state.num_entries and torch.isfinite(host).all(), (i, name)
            logger.info(f"layer {i}: {state.num_entries} compressed entries and index keys, all finite")
        model.release_traced_prefill()
        progress.step("done")
        return

    # --- the first generated token --------------------------------------------------------------- #
    row = logits.reshape(-1)  # already on the host: the last stage streams it over the D2H socket
    assert row.numel() == config.vocab_size and torch.isfinite(row).all(), "non-finite logits"
    top = row.topk(5)
    logger.info(
        "first generated token (top 5): "
        + ", ".join(f"{tokenizer.decode([int(i)])!r} ({v:.2f})" for v, i in zip(top.values.tolist(), top.indices))
    )
    model.release_traced_prefill()
    progress.step("done")
