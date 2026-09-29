# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""End to end on ONE galaxy: the pure-ttnn chunked prefill (``models/demos/deepseek_v3_d_p/tt/v4``) -> hand-off -> this
directory's traced decode, both resident in one process on disjoint chips of the ``(8,4)`` mesh:

* decode: :class:`DeepSeekV4Model` on mesh rows 0-1 (two 1x4 TP4 stages, as in the decode demo; ``DEEPSEEK_V4_DSPARK=0``);
* prefill: ``TtV4PrefillRuntime`` on the ``4x4`` submesh of rows 4-7 (SP 4 over rows, TP 4 over columns, 16 routed experts
  per chip, bf8 experts), eager (a prompt below one chunk is chunk 0, which is always eager);
* hand-off: :mod:`..tt.prefill.handoff_ttnn_prefill` (the prefill's bf16 working state -> the decode buffers, in place).

The prompt's first ``T0 = floor((len - 1) / 128) * 128`` tokens are prefilled; the rest (1..128 tokens) are replayed through
``decode_traced``, whose last step gives the first generated token. Like the hand-off of this directory's own prefill, the
whole conversation must stay below 2048 tokens (no lightning-indexer state is handed over yet).

``DEEPSEEK_V4_E2E_COMPARE=sankar`` (or ``1``) additionally builds this directory's prefill (``DeepSeekV4PrefillModel``) on rows 2-3, prefills the
same ``T0`` tokens and compares the two hand-offs layer by layer (ring, entries, CSA prev_kv / prev_gate, PCC) -- the check that
both prefills describe the same decode state -- before decoding from ours. ``DEEPSEEK_V4_E2E_COMPARE=decode`` instead replays the
same ``T0`` tokens through the DECODE itself, reads its caches back and compares our hand-off against the decode's own state, then
rewinds the session -- a check of our prefill + hand-off against the decode's own arithmetic, independent of any other prefill.

Knobs: ``DEEPSEEK_V4_E2E_PROMPT_LEN`` (1000), ``DEEPSEEK_V4_E2E_TEXT``, ``DEEPSEEK_V4_MAX_NEW_TOKENS`` (128),
``DEEPSEEK_V4_E2E_CHUNK`` (our prefill's chunk, default 2048: one chunk covers the 2048-token limit),
``DEEPSEEK_V4_CACHE_DIR`` (the decode's bf4 cache), ``PREFILL_TTNN_CACHE`` (our prefill's .tensorbin cache root),
``PREFILL_HF_MODEL`` (checkpoint for our prefill; defaults to the decode's snapshot dir).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v4_flash.tests.decode.test_full_model_decode_demo import (
    _CACHE_DIR,
    _DEFAULT_MODEL_DIR,
    _DEFAULT_TEXT,
    _PAGE_BLOCK_SIZE,
    _assert_decode_parallelism,
    _build_rope,
    _checkpoint_available,
    _construct_model,
    _tokenize_chat,
    _traced_max_seq,
)
from models.experimental.deepseek_v4_flash.tests.prefill.test_full_model_prefill_demo import (
    _DEFAULT_PROMPT_FILE,
    _TP_SIZE,
    _Progress,
    _build_prompt_ids,
    _env_int,
)
from models.experimental.deepseek_v4_flash.tt.decode.paged_cache import round_context
from models.experimental.deepseek_v4_flash.tt.prefill.handoff_ttnn_prefill import (
    ALIGNMENT,
    LayerHandoff,
    extract_prefill_handoff,
    load_ttnn_prefill_into_decode,
)
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

_MAX_CONTEXT = 2048  # index_topk * CSA rate: no lightning-indexer state in the hand-off yet
_PREFILL_ROW = 4  # our prefill: the 4x4 submesh of mesh rows 4-7
_COMPARE_ROW = 2  # DEEPSEEK_V4_E2E_COMPARE: this directory's prefill on rows 2-3


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "num_command_queues": 2,
            # our prefill's MoE routing semaphores live in L1_SMALL (DeepSeekV4FlashAdapter.l1_small_size)
            "l1_small_size": 1152,
        }
    ],
    indirect=["device_params"],
    ids=["fabric_2d_torus_xy"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=["mesh_device"], ids=["galaxy_8x4"])
def test_ttnn_prefill_decode_e2e(mesh_device, reset_seeds) -> None:
    from transformers import AutoTokenizer

    progress = _Progress(
        interval=float(os.environ.get("DEEPSEEK_V4_PREFILL_HEARTBEAT", 30)),
        stall=float(os.environ.get("DEEPSEEK_V4_PREFILL_STALL_SECS", 1800)),
    )
    progress.verbose = False
    os.environ["DEEPSEEK_V4_DSPARK"] = "0"  # no MTP link on row 2 (read by DeepSeekV4Model.__init__)
    with progress, contextlib.ExitStack() as prefetcher:
        _run(mesh_device, progress, prefetcher, AutoTokenizer)


def _eos_ids(config) -> set[int]:
    eos = config.eos_token_id
    return {int(eos)} if isinstance(eos, int) else {int(e) for e in (eos or [])}


def _build_ttnn_prefill(mesh_device, chunk: int, max_seq: int, progress: _Progress):
    """Our runtime on the rows-4..7 4x4 submesh, built through the prefill engine's adapter (weights from the .tensorbin
    cache when complete, else dequantised from the checkpoint and cached)."""
    os.environ.setdefault("PREFILL_HF_MODEL", str(DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR).snapshot_dir))
    os.environ["PREFILL_MAX_SEQ_LEN"] = str(max_seq)
    os.environ.setdefault("PREFILL_CSA_SPARSE", "1")
    os.environ.setdefault("PREFILL_MHC_STACKED", "1")
    from models.demos.common.prefill.adapter import PrefillRunParams
    from models.demos.deepseek_v3_d_p.tt.runners.adapters.deepseek_v4_flash import DeepSeekV4FlashAdapter

    sub = mesh_device.create_submesh(ttnn.MeshShape(4, 4), ttnn.MeshCoordinate(_PREFILL_ROW, 0))
    adapter = DeepSeekV4FlashAdapter()
    hf_config = adapter.load_hf_config()
    params = PrefillRunParams(
        mesh_shape=(4, 4),
        num_layers=int(hf_config.num_hidden_layers),
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=max_seq,
        chunk_size=chunk,
        num_users=1,
        capacity_factor=int(os.environ.get("PREFILL_CAPACITY_FACTOR", 8)),
        num_links=int(os.environ.get("PREFILL_NUM_LINKS", 2)),
        gate_mode_name=adapter.default_gate_mode,
        kv_only_last_layer=True,
        weight_cache_path=adapter.weight_cache_path((4, 4)),
        use_trace=False,
    )
    t0 = time.perf_counter()
    rt = adapter.build_runtime(mesh_device=sub, hf_config=hf_config, params=params)
    caches = adapter.allocate_kv_cache(mesh_device=sub, hf_config=hf_config, params=params)
    progress.step("our prefill: compile (warm chunk 0)")
    rt.compile(caches)
    logger.info(
        f"our prefill: built + compiled on the 4x4 submesh (rows {_PREFILL_ROW}-{_PREFILL_ROW + 3}) in "
        f"{time.perf_counter() - t0:.1f}s, weight cache {params.weight_cache_path}"
    )
    return rt, caches, sub


def _run_ttnn_prefill(rt, caches, ids: list[int], chunk: int) -> float:
    """Prefill ``ids`` (length a multiple of 128) chunk by chunk into slot 0; returns the wall seconds."""
    t0 = time.perf_counter()
    for start in range(0, len(ids), chunk):
        part = ids[start : start + chunk]
        x = rt.make_chunk_input(part + [0] * (chunk - len(part)))
        rt.prefill_chunk(x, caches, slot_id=0, actual_start=start, actual_end=start + len(part))
    ttnn.synchronize_device(rt.mesh_device)
    return time.perf_counter() - t0


def _sankar_handoff(mesh_device, config, loader, rope, ids: list[int], decode, progress) -> list[LayerHandoff]:
    """This directory's prefill on rows 2-3 over the same tokens, as LayerHandoffs (the reference for the comparison)."""
    from models.experimental.deepseek_v4_flash.tests.prefill.test_full_model_prefill_demo import _ATTENTION_WEIGHT_DTYPE
    from models.experimental.deepseek_v4_flash.tt.model import plan_layer_placement
    from models.experimental.deepseek_v4_flash.tt.prefill.model import DeepSeekV4PrefillModel
    from models.experimental.deepseek_v4_flash.tt.prefill.weights import checkpoint_expert_provider, checkpoint_weights
    from models.experimental.deepseek_v4_flash.tt.system_config import set_active_system_config
    from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache

    num_layers = decode.num_layers
    set_active_system_config(decode.system_config)
    submeshes = [
        mesh_device.create_submesh(ttnn.MeshShape(1, _TP_SIZE), ttnn.MeshCoordinate(_COMPARE_ROW + i, 0))
        for i in range(2)
    ]
    layer_devices = [submeshes[k] for k in plan_layer_placement(num_layers, 2, 1)]
    cache = WeightCache(os.path.join(_CACHE_DIR, os.path.basename(_DEFAULT_MODEL_DIR))) if _CACHE_DIR else None
    prefill = DeepSeekV4PrefillModel(
        config,
        checkpoint_weights(loader, config, num_layers),
        layer_devices[0],
        rope,
        expert_provider=checkpoint_expert_provider(loader),
        num_layers=num_layers,
        cache=cache,
        weight_dtype=_ATTENTION_WEIGHT_DTYPE,
        expert_dtype=decode.system_config.decode.ttnn_weight_dtype,
        tp_size=_TP_SIZE,
        layer_devices=layer_devices,
        dense_csa=True,
        progress=progress,
    )
    prefill.synchronize("uploads")
    _logits, states = prefill.prefill(torch.tensor(ids, dtype=torch.long).unsqueeze(0), chunk_size=1024)
    out = []
    for li, st in enumerate(states[:num_layers]):
        pdev = prefill.layer_devices[li]
        kind = config.layer_types[li]
        ring = prefill.to_host(st.kv_tail, pdev)[0, 0].float()
        entries = prefill.to_host(st.compressed_kv, pdev)[0, 0].float() if st.compressed_kv is not None else None
        prev_kv = prev_gate = None
        if kind == "compressed_sparse_attention":
            attn = prefill.layers[li].self_attn
            hd = config.head_dim
            prev_kv = prefill.to_host(st.csa_prev_kv, pdev).reshape(attn.rate, hd).float()
            bias = torch.stack([prefill.to_host(b, pdev).reshape(-1)[:hd] for b in attn.c_bias_slots]).float()
            prev_gate = prefill.to_host(st.csa_prev_gate, pdev).reshape(attn.rate, hd).float() - bias
        out.append(LayerHandoff(kind=kind, ring=ring, entries=entries, prev_kv=prev_kv, prev_gate=prev_gate))
    return out


def _decode_replay_handoff(decode, sid: int, ids: list[int], config) -> list[LayerHandoff]:
    """Replay ``ids`` through the decode from position 0 and read its per-layer state back as LayerHandoffs."""
    from models.experimental.deepseek_v4_flash.tt.prefill.handoff_ttnn_prefill import _one_replica

    for pos, tok in enumerate(ids):
        decode.decode_traced(int(tok), pos)
    T, window = len(ids), config.sliding_window
    out: dict[int, LayerHandoff] = {}
    for sm in decode.submeshes_io:
        for li in sm["layers"]:
            kind = config.layer_types[li]
            scache = sm["scaches"][li]
            entries = prev_kv = prev_gate = None
            if kind in ("sliding_attention", "compressed_sparse_attention"):
                kv = _one_replica(scache.kv)[0, 0]
                ring = kv[:window]
                if kind == "compressed_sparse_attention":
                    entries = kv[window : window + T // config.compress_rates[kind]]
                    hd = config.head_dim
                    prev_kv = _one_replica(scache.prev_kv)[:, 0, 0, :hd]
                    prev_gate = _one_replica(scache.prev_gate)[:, 0, 0, :hd]
            else:
                pool = _one_replica(sm["pools"][li])  # [blocks, 1, 32, Dh]
                page_row = decode._require_paged().page_row(sid, kind)[0].tolist()
                rows = window + T // config.compress_rates[kind]
                block = pool.shape[2]
                axis = torch.cat([pool[page_row[b], 0] for b in range(-(-rows // block))], 0)[:rows]
                ring, entries = axis[:window], axis[window:rows]
            out[li] = LayerHandoff(kind=kind, ring=ring, entries=entries, prev_kv=prev_kv, prev_gate=prev_gate)
    return [out[li] for li in range(decode.num_layers)]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    if a.numel() != b.numel():
        return float("nan")
    if torch.allclose(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _compare(ours: list[LayerHandoff], ref: list[LayerHandoff]) -> float:
    worst = 1.0
    for li, (o, r) in enumerate(zip(ours, ref)):
        parts = {"ring": _pcc(o.ring, r.ring)}
        if o.entries is not None:
            n = min(o.entries.shape[0], r.entries.shape[0])
            parts["entries"] = _pcc(o.entries[:n], r.entries[:n])
        if o.prev_kv is not None:
            parts["prev_kv"] = _pcc(o.prev_kv, r.prev_kv)
            parts["prev_gate"] = _pcc(o.prev_gate, r.prev_gate)
        worst = min([worst] + [v for v in parts.values() if not math.isnan(v)])
        logger.info(
            f"hand-off vs reference, layer {li:>2} ({o.kind}): " + " ".join(f"{k} {v:.5f}" for k, v in parts.items())
        )
    return worst


def _run(mesh_device, progress: _Progress, prefetcher: contextlib.ExitStack, AutoTokenizer) -> None:
    max_new = _env_int("DEEPSEEK_V4_MAX_NEW_TOKENS", 128)
    prompt_len = _env_int("DEEPSEEK_V4_E2E_PROMPT_LEN", 1000)
    chunk = _env_int("DEEPSEEK_V4_E2E_CHUNK", 2048)
    compare = os.environ.get("DEEPSEEK_V4_E2E_COMPARE", "0")
    compare = {"1": "sankar"}.get(compare, compare)
    if compare not in ("0", "sankar", "decode"):
        raise ValueError(f"DEEPSEEK_V4_E2E_COMPARE={compare!r}: expected 0 | sankar | decode")

    # --- prompt ----------------------------------------------------------------------------------- #
    progress.step("[1/8] tokenizer and prompt")
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else config.eos_token_id
    if isinstance(pad_id, list):
        pad_id = pad_id[0]
    if "DEEPSEEK_V4_E2E_TEXT" in os.environ:
        text = os.environ["DEEPSEEK_V4_E2E_TEXT"]
        prompt_ids = _tokenize_chat(tokenizer, _DEFAULT_TEXT if text == "decode_demo" else text)
    else:
        prompt_file = Path(os.environ.get("DEEPSEEK_V4_PREFILL_PROMPT", _DEFAULT_PROMPT_FILE))
        prompt_ids, info = _build_prompt_ids(tokenizer, json.loads(prompt_file.read_text())[0], prompt_len)
        logger.info(f"prompt from {prompt_file.name}: {info}")
    real_len = len(prompt_ids)
    aligned = (real_len - 1) // ALIGNMENT * ALIGNMENT  # >= 1 token replayed: decode's last step gives the first token
    tail = real_len - aligned
    if real_len + max_new >= _MAX_CONTEXT:
        raise ValueError(
            f"prompt ({real_len}) + max new tokens ({max_new}) must stay below {_MAX_CONTEXT} (no indexer hand-off)"
        )
    if aligned > chunk * 64:
        raise ValueError(f"prefix {aligned} too long for chunk {chunk}")
    prefill_max_seq = max(chunk, -(-aligned // chunk) * chunk)
    logger.info(
        f"prompt: {real_len} tokens = {aligned} prefilled by the ttnn prefill (chunk {chunk}) + {tail} replayed "
        f"through decode; up to {max_new} new tokens"
    )
    logger.info(f"prompt ends: {tokenizer.decode(prompt_ids[-80:])!r}")

    # --- decode (rows 0-1), static state, session, trace capture --------------------------------- #
    progress.step("[2/8] decode model (mesh rows 0-1)")
    needed = real_len + max_new
    max_seq = round_context(_traced_max_seq(config, needed), set(config.compress_rates.values()), _PAGE_BLOCK_SIZE)
    rope = _build_rope(config, max_seq)
    t0 = time.perf_counter()
    decode, lm_head, loader, config = _construct_model(
        mesh_device, prefetcher, tp_size=_TP_SIZE, system_config=None, loader=loader, config=config
    )
    _assert_decode_parallelism(decode, _TP_SIZE)
    logger.info(f"decode model: {decode.num_layers} layers built in {time.perf_counter() - t0:.1f}s")
    decode.prepare_static_decode(
        rope, max_seq, lm_head=lm_head, num_sessions=1, total_tokens=max_seq, block_size=_PAGE_BLOCK_SIZE
    )
    sid = decode.open_session()
    decode.activate_session(sid)
    progress.step(
        "[3/8] throw-away decode step: captures the traces (its cache writes are overwritten by the hand-off)"
    )
    t0 = time.perf_counter()
    decode.decode_traced(pad_id, 0)
    logger.info(f"trace capture + first step: {time.perf_counter() - t0:.1f}s")

    # --- our prefill (rows 4-7) ------------------------------------------------------------------- #
    progress.step(f"[4/8] ttnn prefill on the 4x4 submesh (rows {_PREFILL_ROW}-{_PREFILL_ROW + 3})")
    rt, caches, _sub = _build_ttnn_prefill(mesh_device, chunk, prefill_max_seq, progress)
    progress.step(f"[5/8] ttnn prefill of {aligned} tokens")
    ttft_prefill = _run_ttnn_prefill(rt, caches, prompt_ids[:aligned], chunk)
    logger.info(f"ttnn prefill: {aligned} tokens in {ttft_prefill:.2f}s ({aligned / ttft_prefill:.1f} tok/s)")

    # --- hand-off ----------------------------------------------------------------------------------- #
    progress.step("[6/8] hand-off: ttnn prefill state -> decode buffers")
    t0 = time.perf_counter()
    layers = extract_prefill_handoff(rt, 0, aligned)
    t_extract = time.perf_counter() - t0
    if compare == "sankar":
        progress.step("[6b] reference: this directory's prefill on rows 2-3 over the same tokens")
        ref = _sankar_handoff(mesh_device, config, loader, rope, prompt_ids[:aligned], decode, progress)
        worst = _compare(layers, ref)
        logger.info(f"hand-off comparison vs this directory's prefill: worst PCC {worst:.5f} over {len(layers)} layers")
    elif compare == "decode":
        progress.step(f"[6b] reference: the decode's own state after replaying the same {aligned} tokens")
        t_ref = time.perf_counter()
        ref = _decode_replay_handoff(decode, sid, prompt_ids[:aligned], config)
        logger.info(f"decode replay of {aligned} tokens: {time.perf_counter() - t_ref:.1f}s")
        worst = _compare(layers, ref)
        logger.info(f"hand-off comparison vs the decode's own state: worst PCC {worst:.5f} over {len(layers)} layers")
        decode.reset_session(sid)  # rewind: the hand-off below must land on a clean session
        decode.reset_static_caches()
    t0 = time.perf_counter()
    handed = load_ttnn_prefill_into_decode(
        decode, layers, aligned, sid, progress=lambda m: progress(m, important=False)
    )
    assert handed == aligned
    logger.info(f"hand-off of {handed} tokens: extract {t_extract:.2f}s + load {time.perf_counter() - t0:.2f}s")

    # --- tail replay + generation ---------------------------------------------------------------- #
    progress.step(f"[7/8] replaying {tail} prompt token(s) through decode")
    t0 = time.perf_counter()
    next_id = None
    for pos in range(aligned, real_len):
        logits = decode.decode_traced(prompt_ids[pos], pos).reshape(1, -1).float()
        next_id = int(logits[0].argmax())
    logger.info(
        f"tail replay: {tail} tokens in {time.perf_counter() - t0:.2f}s; first token {next_id} "
        f"{tokenizer.decode([next_id])!r}"
    )
    progress.step(f"[8/8] decode: up to {max_new} tokens")
    eos = _eos_ids(config)
    generated = [next_id]
    step_times: list[float] = []
    for step in range(1, max_new):
        if generated[-1] in eos:
            logger.info("hit EOS; stopping")
            break
        t0 = time.perf_counter()
        logits = decode.decode_traced(generated[-1], real_len + step - 1).reshape(1, -1).float()
        generated.append(int(logits[0].argmax()))
        step_times.append(time.perf_counter() - t0)
    if step_times:
        logger.info(f"decode: {len(generated)} tokens, {len(step_times) / sum(step_times):.2f} tok/s steady")
    logger.info("PROMPT (last 600 chars):\n" + tokenizer.decode(prompt_ids)[-600:])
    logger.info(f"GENERATED ({len(generated)} tokens):\n{tokenizer.decode(generated)}")
    logger.info(
        f"E2E SUMMARY: prompt {real_len} (prefilled {aligned}, tail {tail}) | ttnn prefill {ttft_prefill:.2f}s | "
        f"generated {len(generated)} tokens"
    )
    rt.release_trace()
    assert generated and all(0 <= t < config.vocab_size for t in generated)
