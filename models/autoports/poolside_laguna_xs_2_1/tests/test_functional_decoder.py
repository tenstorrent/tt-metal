# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Functional-decoder correctness tests for poolside/Laguna-XS-2.1.

Covers every meaningful layer kind at REAL target config shapes with REAL weights:
  * layer 0  – full_attention + dense MLP   (48 heads, YARN partial rope rd=64)
  * layer 1  – sliding_attention + MoE      (64 heads, default rope rd=128, win=512)
  * layer 4  – full_attention + MoE         (48 heads, YARN partial rope rd=64)

Validates paged prefill, paged decode, page-table handling (permuted / nonzero
slots), tensor current-position, non-aligned logical lengths around tile/window
boundaries, batch>1, multi-step decode, determinism, and full advertised context
(262144) decode via an identically-seeded KV cache.

ENVIRONMENT: run against the installed ttnn tree from a non-repo cwd so tt-metal
does not auto-detect the dev git root (which triggers a JIT header/source
mismatch):
    python -m pytest models/autoports/poolside_laguna_xs_2_1/tests/test_functional_decoder.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W
from models.autoports.poolside_laguna_xs_2_1.tt.functional_decoder import FunctionalDecoder

PCC_BAR = 0.995
HIDDEN = 2048

# layer kinds under test
FULL_DENSE = 0
SLIDING_MOE = 1
FULL_MOE = 4
ALL_LAYERS = [FULL_DENSE, SLIDING_MOE, FULL_MOE]


def _pcc(a, b):
    a = a.flatten().float().numpy()
    b = b.flatten().float().numpy()
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# Session-scoped device + per-layer build caches
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    yield dev
    ttnn.close_mesh_device(dev)


@pytest.fixture(scope="module")
def hf_config():
    return R.build_config()


_CTX = {}
_RAW = {}


def _ctx(hf_config, layer):
    if layer not in _CTX:
        raw = W.load_layer_tensors(layer)
        _RAW[layer] = raw
        _CTX[layer] = R.make_context(
            hf_config, layer, state_dict=W.to_hf_layer_state_dict(raw, hf_config, layer), dtype=torch.float32
        )
    return _CTX[layer], _RAW[layer]


_DEC = {}


def _decoder(hf_config, layer, device, max_seq_len=None):
    # Build ONE decoder per layer at full advertised context and reuse it. The RoPE
    # table is cheap (~64 MB); caching a separate full-expert-weight decoder per
    # max_seq_len would exhaust the ~34 GB device DRAM. ``max_seq_len`` is ignored;
    # per-test KV caches are sized independently in alloc_kv_cache.
    if layer not in _DEC:
        _, raw = _ctx(hf_config, layer)
        _DEC[layer] = FunctionalDecoder.from_state_dict(
            raw, hf_config=hf_config, layer_idx=layer, mesh_device=device, max_seq_len=hf_config.max_position_embeddings
        )
    return _DEC[layer]


def _tt(x, device):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


# --------------------------------------------------------------------------- #
# Prefill PCC — smoke, tile boundary, window boundary, just-over, non-aligned
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seq", [8, 32, 100, 512, 513, 1024, 2048])
@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_prefill_pcc(device, hf_config, layer, seq):
    ctx, raw = _ctx(hf_config, layer)
    max_seq = max(seq + 32, 256)
    dec = _decoder(hf_config, layer, device, max_seq)
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=max_seq, block_size=32)
    pt = dec.make_page_table(1, kv["blocks_per_user"])
    torch.manual_seed(seq)
    x = torch.randn(1, seq, HIDDEN) * 0.5
    ref, _ = R.reference_forward(ctx, x)
    out = dec.prefill_forward(_tt(x, device), kv, pt, user_id=0, start_pos=0)
    got = ttnn.to_torch(out).float().reshape(1, seq, HIDDEN)
    pcc = _pcc(got, ref)
    assert pcc >= PCC_BAR, f"layer {layer} prefill seq={seq} PCC {pcc:.5f} < {PCC_BAR}"


# --------------------------------------------------------------------------- #
# Decode PCC after a prefill (paged decode + tensor current-position)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prefill_seq", [32, 513, 2048])
@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_decode_pcc(device, hf_config, layer, prefill_seq):
    ctx, raw = _ctx(hf_config, layer)
    max_seq = prefill_seq + 8
    dec = _decoder(hf_config, layer, device, max_seq)
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=max_seq, block_size=32)
    pt = dec.make_page_table(1, kv["blocks_per_user"])
    torch.manual_seed(prefill_seq)
    x = torch.randn(1, prefill_seq, HIDDEN) * 0.5
    _, pkv = R.reference_forward(ctx, x)
    dec.prefill_forward(_tt(x, device), kv, pt, user_id=0, start_pos=0)
    xd = torch.randn(1, 1, HIDDEN) * 0.5
    ref, _ = R.reference_forward(ctx, xd, past_key_values=pkv)
    cur = ttnn.from_torch(
        torch.tensor([prefill_seq], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    ridx = ttnn.from_torch(
        torch.tensor([[prefill_seq]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    out = dec.decode_forward(_tt(xd.reshape(1, 1, 1, HIDDEN), device), cur, ridx, pt, kv)
    got = ttnn.to_torch(out).float().reshape(1, 1, HIDDEN)
    pcc = _pcc(got, ref)
    assert pcc >= PCC_BAR, f"layer {layer} decode pos={prefill_seq} PCC {pcc:.5f} < {PCC_BAR}"


# --------------------------------------------------------------------------- #
# Multi-step sequential decode: prefill S, then decode several tokens
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_multistep_decode(device, hf_config, layer):
    ctx, raw = _ctx(hf_config, layer)
    S, steps = 48, 5
    max_seq = S + steps + 8
    dec = _decoder(hf_config, layer, device, max_seq)
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=max_seq, block_size=32)
    pt = dec.make_page_table(1, kv["blocks_per_user"])
    torch.manual_seed(7)
    x = torch.randn(1, S, HIDDEN) * 0.5
    _, pkv = R.reference_forward(ctx, x)
    dec.prefill_forward(_tt(x, device), kv, pt, user_id=0, start_pos=0)
    for t in range(steps):
        pos = S + t
        xd = torch.randn(1, 1, HIDDEN) * 0.5
        ref, _ = R.reference_forward(ctx, xd, past_key_values=pkv)
        cur = ttnn.from_torch(
            torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        ridx = ttnn.from_torch(
            torch.tensor([[pos]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        out = dec.decode_forward(_tt(xd.reshape(1, 1, 1, HIDDEN), device), cur, ridx, pt, kv)
        got = ttnn.to_torch(out).float().reshape(1, 1, HIDDEN)
        pcc = _pcc(got, ref)
        assert pcc >= PCC_BAR, f"layer {layer} step {t} pos={pos} PCC {pcc:.5f}"


# --------------------------------------------------------------------------- #
# Batch>1 prefill + decode with a permuted / nonzero page table
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("B", [4, 32])
@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_batch_prefill_decode(device, hf_config, layer, B):
    ctx, raw = _ctx(hf_config, layer)
    S = 48
    max_seq = S + 8
    dec = _decoder(hf_config, layer, device, max_seq)
    kv = dec.alloc_kv_cache(max_users=B, max_seq_len=max_seq, block_size=32)
    bpu = kv["blocks_per_user"]
    torch.manual_seed(3)
    perm = torch.randperm(B * bpu).to(torch.int32).reshape(B, bpu)
    pt = ttnn.from_torch(perm, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    xs = [torch.randn(1, S, HIDDEN) * 0.5 for _ in range(B)]
    pkvs = []
    for u in range(B):
        ref, pkv = R.reference_forward(ctx, xs[u])
        pkvs.append(pkv)
        out = dec.prefill_forward(_tt(xs[u], device), kv, pt, user_id=u, start_pos=0)
        got = ttnn.to_torch(out).float().reshape(1, S, HIDDEN)
        assert _pcc(got, ref) >= PCC_BAR, f"layer {layer} user {u} prefill PCC low"
    xd = torch.randn(1, 1, B, HIDDEN) * 0.5
    refs = [R.reference_forward(ctx, xd[:, :, u, :], past_key_values=pkvs[u])[0] for u in range(B)]
    ref_dec = torch.cat(refs, dim=1).reshape(B, HIDDEN)
    cur = ttnn.from_torch(
        torch.full((B,), S, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    ridx = ttnn.from_torch(
        torch.full((1, B), S, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    out = dec.decode_forward(_tt(xd, device), cur, ridx, pt, kv)
    got = ttnn.to_torch(out).float().reshape(B, HIDDEN)
    pcc = _pcc(got, ref_dec)
    assert pcc >= PCC_BAR, f"layer {layer} batch decode PCC {pcc:.5f}"


# --------------------------------------------------------------------------- #
# Determinism: identical inputs -> identical outputs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_determinism(device, hf_config, layer):
    _ctx(hf_config, layer)
    S = 64
    dec = _decoder(hf_config, layer, device, S + 8)

    def run():
        kv = dec.alloc_kv_cache(max_users=1, max_seq_len=S + 8, block_size=32)
        pt = dec.make_page_table(1, kv["blocks_per_user"])
        torch.manual_seed(11)
        x = torch.randn(1, S, HIDDEN) * 0.5
        dec.prefill_forward(_tt(x, device), kv, pt, user_id=0, start_pos=0)
        xd = torch.randn(1, 1, HIDDEN) * 0.5
        cur = ttnn.from_torch(
            torch.tensor([S], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        ridx = ttnn.from_torch(
            torch.tensor([[S]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        out = dec.decode_forward(_tt(xd.reshape(1, 1, 1, HIDDEN), device), cur, ridx, pt, kv)
        return ttnn.to_torch(out).float()

    a = run()
    b = run()
    assert torch.equal(a, b), f"layer {layer} decode not deterministic (max diff {(a - b).abs().max()})"


# --------------------------------------------------------------------------- #
# Full advertised context (262144) decode via identically-seeded KV cache
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_decode_traced_pcc(device, hf_config, layer):
    """TRACED decode correctness: capture the decode graph once, then replay it over several
    steps updating the input / cur_pos / rope_idx device tensors IN PLACE (never reallocating,
    so baked buffer addresses stay valid), and compare each replayed output to the HF
    reference. This proves the traced-execution path (not just an eager forward) is correct,
    that current-position propagates through the trace, and that there is no stale-input hazard."""
    ctx, raw = _ctx(hf_config, layer)
    S, steps = 48, 4
    dec = _decoder(hf_config, layer, device)
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=S + steps + 8, block_size=32)
    pt = dec.make_page_table(1, kv["blocks_per_user"])
    torch.manual_seed(123)
    x = torch.randn(1, S, HIDDEN) * 0.5
    _, pkv = R.reference_forward(ctx, x)
    dec.prefill_forward(_tt(x, device), kv, pt, user_id=0, start_pos=0)  # seed cache (eager)

    # persistent trace-input device tensors (allocated once)
    def host_x(t):
        return ttnn.from_torch(t.reshape(1, 1, 1, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def host_pos(p):
        return ttnn.from_torch(torch.tensor([p], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def host_ridx(p):
        return ttnn.from_torch(torch.tensor([[p]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    step_x = [torch.randn(1, 1, HIDDEN) * 0.5 for _ in range(steps)]
    x_dev = _tt(step_x[0].reshape(1, 1, 1, HIDDEN), device)
    cur_dev = ttnn.from_torch(
        torch.tensor([S], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    ridx_dev = ttnn.from_torch(
        torch.tensor([[S]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    # compile once (eager) so the trace only captures a warmed program
    dec.decode_forward(x_dev, cur_dev, ridx_dev, pt, kv)
    ttnn.synchronize_device(device)
    # capture
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_dev = dec.decode_forward(x_dev, cur_dev, ridx_dev, pt, kv)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    try:
        for t in range(steps):
            pos = S + t
            # update the SAME device tensors in place (no realloc)
            ttnn.copy_host_to_device_tensor(host_x(step_x[t]), x_dev)
            ttnn.copy_host_to_device_tensor(host_pos(pos), cur_dev)
            ttnn.copy_host_to_device_tensor(host_ridx(pos), ridx_dev)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            got = ttnn.to_torch(out_dev).float().reshape(1, 1, HIDDEN)
            ref, _ = R.reference_forward(ctx, step_x[t], past_key_values=pkv)
            pcc = _pcc(got, ref)
            assert pcc >= PCC_BAR, f"layer {layer} TRACED decode step {t} pos={pos} PCC {pcc:.5f} < {PCC_BAR}"
    finally:
        ttnn.release_trace(device, tid)


@pytest.mark.parametrize("seq", [1500, 2048])
@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_prefill_chunked_matches_hf(device, hf_config, layer, seq):
    """Force Q-chunking (small PREFILL_SDPA_CHUNK) and confirm the chunked prefill path
    (paged chunked SDPA for full layers; overlapping local slices for sliding) reproduces
    HF, incl. non-128-aligned lengths. This is the path used for prompts beyond the
    single-shot SDPA op limit (32768), so the full advertised context prefills correctly."""
    ctx, raw = _ctx(hf_config, layer)
    dec = _decoder(hf_config, layer, device)
    orig = dec.PREFILL_SDPA_CHUNK
    dec.PREFILL_SDPA_CHUNK = 512
    try:
        kv = dec.alloc_kv_cache(max_users=1, max_seq_len=seq + 32, block_size=32)
        pt = dec.make_page_table(1, kv["blocks_per_user"])
        torch.manual_seed(seq)
        x = torch.randn(1, seq, HIDDEN) * 0.5
        ref, _ = R.reference_forward(ctx, x)
        out = dec.prefill_forward(_tt(x, device), kv, pt, user_id=0, start_pos=0)
        got = ttnn.to_torch(out).float().reshape(1, seq, HIDDEN)
        pcc = _pcc(got, ref)
        assert pcc >= PCC_BAR, f"layer {layer} chunked prefill seq={seq} PCC {pcc:.5f} < {PCC_BAR}"
    finally:
        dec.PREFILL_SDPA_CHUNK = orig


@pytest.mark.parametrize("seq", [1500, 2048])
@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_prefill_pipelined_matches_hf(device, hf_config, layer, seq):
    """Force sequence-pipelined prefill (small PIPE_CHUNK) and confirm it reproduces HF.
    This is the path that lets prompts exceeding single-shot DRAM (up to the full 262144
    context) prefill within device memory; capacity at 262144 is recorded in
    doc/functional_decoder/prefill_capacity.json (both layer kinds, finite output)."""
    ctx, raw = _ctx(hf_config, layer)
    dec = _decoder(hf_config, layer, device)
    orig = dec.PIPE_CHUNK
    dec.PIPE_CHUNK = 512
    try:
        kv = dec.alloc_kv_cache(max_users=1, max_seq_len=seq + 32, block_size=32)
        pt = dec.make_page_table(1, kv["blocks_per_user"])
        torch.manual_seed(seq + 1)
        x = torch.randn(1, seq, HIDDEN) * 0.5
        ref, _ = R.reference_forward(ctx, x)
        out = dec.prefill_forward(_tt(x, device), kv, pt, user_id=0, start_pos=0)
        got = ttnn.to_torch(out).float().reshape(1, seq, HIDDEN)
        pcc = _pcc(got, ref)
        assert pcc >= PCC_BAR, f"layer {layer} pipelined prefill seq={seq} PCC {pcc:.5f} < {PCC_BAR}"
    finally:
        dec.PIPE_CHUNK = orig


@pytest.mark.parametrize("layer", [FULL_MOE, SLIDING_MOE])
def test_full_context_decode(device, hf_config, layer):
    ctx, raw = _ctx(hf_config, layer)
    n_ctx = hf_config.max_position_embeddings - 1  # 262143 seeded, decode at 262143
    max_seq = hf_config.max_position_embeddings
    dec = _decoder(hf_config, layer, device, max_seq)
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=max_seq, block_size=32)
    pt = dec.make_page_table(1, kv["blocks_per_user"])
    # seed identical K/V into paged + HF caches (single contiguous fill)
    from transformers.cache_utils import DynamicCache

    cfg = dec.cfg
    g = torch.Generator().manual_seed(99)
    # Validate full-context robustly without conflating it with the softmax-weighting bf16
    # floor: seed VARIED keys (the query's score path runs over ALL 262144 cached positions,
    # and its RoPE is evaluated at absolute position 262143) but ZERO cached values. The
    # attention output is then w_self * v_new (only the freshly-decoded token has non-zero V),
    # whose DIRECTION is v_new in both bf16 and fp32 — so PCC is invariant to the exact
    # softmax weight and robust to the bf16-vs-fp32 accumulation over 262144 terms. This
    # exercises what the full-context test must prove: cache fill/read, page-table over 8192
    # blocks, and int32 cur_pos + RoPE addressing at the advertised 262144. The attention
    # VALUE-weighting numerics are validated separately at feasible lengths with real
    # weights/inputs (test_decode_pcc: decode PCC 0.9997).
    k = (torch.randn(1, cfg.num_kv_heads, n_ctx, cfg.head_dim, generator=g)).to(torch.bfloat16).float()
    v = torch.zeros(1, cfg.num_kv_heads, n_ctx, cfg.head_dim)
    ttnn.experimental.paged_fill_cache(kv["k"], _tt(k, device), pt, batch_idx=0)
    ttnn.experimental.paged_fill_cache(kv["v"], _tt(v, device), pt, batch_idx=0)
    pkv = DynamicCache(config=ctx.config)
    pkv.update(k.clone(), v.clone(), 0)
    # decode one token at the last context position
    xd = torch.randn(1, 1, HIDDEN) * 0.5
    ref, _ = R.reference_forward(ctx, xd, past_key_values=pkv)
    cur = ttnn.from_torch(
        torch.tensor([n_ctx], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    ridx = ttnn.from_torch(
        torch.tensor([[n_ctx]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    out = dec.decode_forward(_tt(xd.reshape(1, 1, 1, HIDDEN), device), cur, ridx, pt, kv)
    got = ttnn.to_torch(out).float().reshape(1, 1, HIDDEN)
    assert torch.isfinite(got).all(), f"layer {layer} full-context output not finite"
    pcc = _pcc(got, ref)
    assert pcc >= PCC_BAR, f"layer {layer} full-context (pos={n_ctx}) decode PCC {pcc:.5f} < {PCC_BAR}"
