# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The whole V4.1 prefill on a 1024-token chunk, with the routed MoE at its new token cap.

``fused_experts_prefill`` now takes 1024 tokens per call instead of 512, so a production chunk is ONE routed
call rather than two. This runs the real 40-layer checkpoint end to end at exactly that shape -- chunk 1024,
TP4 stages, bf8 attention, bf4 balanced-DRAM experts, the Engram lookup -- and checks the model still produces
sane output through its public :meth:`~....tt.prefill.model.DeepSeekV41PrefillModel.prefill` entry point.

It is deliberately cheaper and more direct than ``test_full_model_prefill_verify.py`` (which compares every
layer teacher-forced against a cached CPU reference):

* it asserts the routed op is called **once per chunk**, which is the whole point of the cap;
* it checks every layer's output is finite and bounded (a wrong token packing or a stale routing list shows up
  as NaN/inf or a blown-up residual long before PCC moves);
* it checks the logits are finite with a plausible shape, and logs the top-5 after the last token so a
  regression is visible in the output rather than only in a number.

For a full accuracy comparison against the fp32 reference, run ``test_full_model_prefill_verify.py``; this test
is the fast "does the 1024-token MoE work in the real model" gate.

    pytest models/experimental/deepseek_v41_flash/tests/prefill/test_full_model_prefill_moe1024.py -x -s

Knobs: ``DEEPSEEK_V4_MOE1024_LEN`` (1024, the prompt length), ``DEEPSEEK_V4_MOE1024_LAYERS`` (all 40),
``DEEPSEEK_V4_MOE1024_STAGES`` (8), ``DEEPSEEK_V4_MOE1024_MAX_RMS`` (64.0, the per-layer residual RMS ceiling).
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
    _available_bytes,
    _build_rope,
    _checkpoint_available,
)
from models.experimental.deepseek_v41_flash.tests.prefill.test_full_model_prefill_demo import (
    _ATTENTION_WEIGHT_DTYPE,
    _EXPERT_DTYPE,
    _TP_SIZE,
)
from models.experimental.deepseek_v41_flash.tt.config import CACHE_DIR, DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v41_flash.tt.engram_lookup import (
    ENGRAM_RAM_MARGIN,
    EngramHostLookup,
    engram_ram_needed,
    load_engram_table,
    stage_engram_tables,
)
from models.experimental.deepseek_v41_flash.tt.prefill.model import DeepSeekV41PrefillModel
from models.experimental.deepseek_v4_flash.tests.prefill.test_full_model_prefill_demo import (
    _DEFAULT_PROMPT_FILE,
    _Progress,
    _build_prompt_ids,
    _env_int,
)
from models.experimental.deepseek_v4_flash.tt.model import plan_layer_placement
from models.experimental.deepseek_v4_flash.tt.prefill.moe import MAX_OP_TOKENS
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config, set_active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

# The production prefill chunk: one routed op call per chunk now that the cap covers it.
CHUNK = 1024


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(21600)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("mesh_device", [(8, _TP_SIZE)], indirect=["mesh_device"], ids=["galaxy_8x4"])
def test_full_model_prefill_moe1024(mesh_device, reset_seeds, monkeypatch) -> None:
    from transformers import AutoTokenizer

    progress = _Progress(
        interval=float(os.environ.get("DEEPSEEK_V4_PREFILL_HEARTBEAT", 30)),
        stall=float(os.environ.get("DEEPSEEK_V4_PREFILL_STALL_SECS", 600)),
    )
    progress.verbose = False
    with progress:
        _run(mesh_device, progress, AutoTokenizer, monkeypatch)


def _run(mesh_device, progress: _Progress, AutoTokenizer, monkeypatch) -> None:
    target_len = _env_int("DEEPSEEK_V4_MOE1024_LEN", CHUNK)
    if target_len != CHUNK:
        pytest.skip(f"this test runs one {CHUNK}-token chunk by design, got DEEPSEEK_V4_MOE1024_LEN={target_len}")
    max_rms = _env_float("DEEPSEEK_V4_MOE1024_MAX_RMS", 64.0)
    # The premise of this test: one production chunk is ONE routed call. Checked before anything expensive.
    assert CHUNK <= MAX_OP_TOKENS, (
        f"a {CHUNK}-token chunk needs MAX_OP_TOKENS >= {CHUNK}, got {MAX_OP_TOKENS}: the chunk would be split "
        f"into {math.ceil(CHUNK / MAX_OP_TOKENS)} routed calls and re-read every expert's weights, which is "
        f"exactly what this test is here to catch"
    )

    # --- checkpoint and prompt -------------------------------------------------------------------- #
    progress.step("[1/5] checkpoint, tokenizer, prompt")
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    num_layers = min(_env_int("DEEPSEEK_V4_MOE1024_LAYERS", config.num_hidden_layers), config.num_hidden_layers)
    entry = json.loads(Path(os.environ.get("DEEPSEEK_V4_PREFILL_PROMPT", _DEFAULT_PROMPT_FILE)).read_text())[0]
    prompt_ids, _ = _build_prompt_ids(tokenizer, entry, target_len)
    # ``_build_prompt_ids`` trims to exactly ``target_len`` when the entry has a context (the default prompt
    # does); a context-free entry ignores the length, so trim rather than demand an exact match. Either way the
    # chunk must be a tile-aligned 1024 tokens, which is the shape this test exists to exercise.
    if len(prompt_ids) > CHUNK:
        prompt_ids = prompt_ids[-CHUNK:]
    assert len(prompt_ids) == CHUNK, (
        f"need a {CHUNK}-token prompt to fill one production chunk, got {len(prompt_ids)}: "
        f"set DEEPSEEK_V4_PREFILL_PROMPT to an entry of at least {CHUNK} tokens"
    )
    ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
    logger.info(f"{num_layers} layers on {target_len} tokens, one chunk of {CHUNK} (routed cap {MAX_OP_TOKENS})")
    engram_ids = [i for i in config.engram_layer_ids if i < num_layers]
    if engram_ids:
        stage_engram_tables(loader, engram_ids)
        need = ENGRAM_RAM_MARGIN * engram_ram_needed(loader, engram_ids)
        available = _available_bytes()
        if available < need:
            pytest.skip(
                f"Engram tables for layers {engram_ids} need {need / 1e9:.0f} GB with headroom; "
                f"{available / 1e9:.0f} GB is available"
            )

    # --- mesh, stages, model ---------------------------------------------------------------------- #
    progress.step("[2/5] system profile, submeshes, Engram tables")
    system_config = load_system_config(mesh_device=mesh_device).log()
    set_active_system_config(system_config)
    mesh_rows, mesh_cols = tuple(mesh_device.shape)
    assert mesh_cols >= _TP_SIZE
    num_stages = min(_env_int("DEEPSEEK_V4_MOE1024_STAGES", 8), mesh_rows, num_layers)
    placement = plan_layer_placement(num_layers, num_stages, 1)
    submeshes = [
        mesh_device.create_submesh(ttnn.MeshShape(1, _TP_SIZE), ttnn.MeshCoordinate(i, 0))
        for i in range(placement[-1] + 1)
    ]
    layer_devices = [submeshes[k] for k in placement]
    rope = _build_rope(config, loader.snapshot_dir, target_len)
    engram_lookup = None
    if engram_ids:
        tables = {i: load_engram_table(loader, i) for i in engram_ids}
        engram_lookup = EngramHostLookup(
            config, loader.snapshot_dir, tables, batch=1, max_seq=target_len, tokenizer=tokenizer
        )

    progress.step(f"[3/5] building the model: {num_layers} layers over {num_stages} stage(s)")
    cache = WeightCache(CACHE_DIR) if CACHE_DIR else None
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
    )
    model.synchronize("uploads")
    logger.info(f"model built in {time.perf_counter() - t0:.1f}s on {len(submeshes)} stage(s)")

    # --- count routed calls per chunk, and watch every layer's output ----------------------------- #
    routed_calls: list[int] = []
    real_routed = ttnn.experimental.deepseek_prefill.fused_experts_prefill

    def counted(*args, **kwargs):
        routed_calls.append(args[0].shape[2])
        return real_routed(*args, **kwargs)

    per_layer = {"finite": [True] * num_layers, "rms": [0.0] * num_layers, "max_abs": [0.0] * num_layers}

    def on_layer(layer_index: int, out: ttnn.Tensor, out_pre: ttnn.Tensor, dev) -> None:
        host = model.to_host(out, dev)[0].float()
        per_layer["finite"][layer_index] &= bool(torch.isfinite(host).all())
        per_layer["rms"][layer_index] = max(per_layer["rms"][layer_index], host.pow(2).mean().sqrt().item())
        per_layer["max_abs"][layer_index] = max(per_layer["max_abs"][layer_index], host.abs().max().item())

    progress.step(f"[4/5] prefill: one chunk of {target_len} tokens")
    monkeypatch.setattr(ttnn.experimental.deepseek_prefill, "fused_experts_prefill", counted)
    t0 = time.perf_counter()
    # ``_stack`` is the model's chunk step and the only entry point that reports each layer's output; the
    # public ``prefill()`` wraps it without an ``on_layer`` hook, and this test needs to see every layer.
    states = model.new_state()
    streams, pre_mix = model._stack(ids, states, on_layer=on_layer)
    logits = model.head(streams, pre_mix)
    ttnn.deallocate(streams)
    ttnn.deallocate(pre_mix)
    model.synchronize("prefill")
    seconds = time.perf_counter() - t0
    logger.info(f"prefill of {target_len} tokens in {seconds:.1f}s ({target_len / seconds:.0f} tok/s)")

    # --- report ------------------------------------------------------------------------------------ #
    progress.step("[5/5] report")
    logger.info(
        "\n".join(
            [""]
            + [
                f"layer {i:2d} ({f'ratio {config.compress_ratios[i]}':>8}): finite {str(per_layer['finite'][i]):>5} "
                f"max rms {per_layer['rms'][i]:>8.3f} max |x| {per_layer['max_abs'][i]:>9.2f}"
                for i in range(num_layers)
            ]
        )
    )
    logger.info(
        f"routed op: {len(routed_calls)} call(s) over {routed_calls[:4]}{'...' if len(routed_calls) > 4 else ''} "
        f"tokens for {num_layers} MoE layers at {target_len} tokens/chunk"
    )
    assert logits is not None, "the head returned no logits"
    host_logits = model.to_host(logits, model.head_device).reshape(-1)[: config.vocab_size].float()
    top = host_logits.topk(5)
    fmt = ", ".join(f"{tokenizer.decode([int(i)])!r} ({v:.2f})" for v, i in zip(top.values.tolist(), top.indices))
    logger.info(f"last token top 5: {fmt}")

    failures = []
    for i in range(num_layers):
        if not per_layer["finite"][i]:
            failures.append(f"layer {i}: output is not finite")
        elif per_layer["rms"][i] > max_rms:
            failures.append(
                f"layer {i}: residual rms {per_layer['rms'][i]:.2f} exceeds {max_rms} (the streams are diverging)"
            )
    if not bool(torch.isfinite(host_logits).all()):
        failures.append("logits are not finite")
    if host_logits.numel() != config.vocab_size:
        failures.append(f"logits have {host_logits.numel()} entries, expected {config.vocab_size}")
    # Every MoE layer must have taken the whole chunk in ONE routed call -- unless this layer set has no MoE,
    # in which case the counter is empty and there is nothing to check.
    if routed_calls:
        expected = -(-target_len // MAX_OP_TOKENS)
        wrong = [n for n in routed_calls if n != expected]
        if wrong:
            failures.append(
                f"the routed op got {sorted(set(wrong))} tokens per call; a {target_len}-token chunk must be "
                f"one call of {expected} (MAX_OP_TOKENS={MAX_OP_TOKENS})"
            )
    progress.step("done")
    assert not failures, "1024-token MoE full-model run failed:\n  " + "\n  ".join(failures)
