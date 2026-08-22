# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the OPTIMIZED Laguna-XS-2.1 decoder.

Mirrors tests/test_functional_decoder.py (same layer-only HF reference, same PCC bar
0.995, same real weights for the three meaningful layer kinds) but drives
OptimizedDecoder — so these tests exercise the OPTIMIZED path (packed QKV, BFP8/BFP4
weights, LoFi/HiFi2 fidelity, DRAM-sharded decode matmuls, BFP8 paged KV cache,
L1 sparse-expert outputs), not a functional fallback.

Coverage: paged prefill/decode, page-table (permuted/nonzero), tensor current-position,
non-aligned logical lengths (tile/window boundaries), batch up to 32, multi-step decode,
determinism, TRACED decode replay, forced-chunk / sequence-pipelined long prefill,
full advertised context (262144) decode addressing, repeated-run stress, and an assertion
that the measured path is the optimized (non-fallback) one.

ENVIRONMENT (installed self-consistent tree from a non-repo cwd; see functional README):
    python -m pytest models/autoports/poolside_laguna_xs_2_1/tests/test_optimized_decoder.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W
from models.autoports.poolside_laguna_xs_2_1.tt.optimized_decoder import OptimizedDecoder

PCC_BAR = 0.995
HIDDEN = 2048

FULL_DENSE = 0
SLIDING_MOE = 1
FULL_MOE = 4
ALL_LAYERS = [FULL_DENSE, SLIDING_MOE, FULL_MOE]


def _pcc(a, b):
    a = a.flatten().float().numpy()
    b = b.flatten().float().numpy()
    return float(np.corrcoef(a, b)[0, 1])


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
    # One optimized decoder per layer at full advertised context, reused (RoPE table
    # cheap; a full-expert-weight decoder per max_seq_len would exhaust device DRAM).
    if layer not in _DEC:
        _, raw = _ctx(hf_config, layer)
        _DEC[layer] = OptimizedDecoder.from_state_dict(
            raw, hf_config=hf_config, layer_idx=layer, mesh_device=device, max_seq_len=hf_config.max_position_embeddings
        )
    return _DEC[layer]


def _tt(x, device):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


# --------------------------------------------------------------------------- #
# The measured path must be the OPTIMIZED one (not a functional fallback).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_optimized_path_active(device, hf_config, layer):
    dec = _decoder(hf_config, layer, device)
    assert dec.use_dram_sharded, "DRAM-sharded decode matmuls must be enabled (optimized path)"
    # BFP8 paged KV cache + DRAM-width-sharded weight copies present
    assert dec.policy.kv_cache == ttnn.bfloat8_b
    assert "wqkv_ds" in dec.w and "wo_ds" in dec.w, "packed-QKV/O DRAM-sharded decode weights missing"
    assert dec.w["wqkv"].dtype == ttnn.bfloat8_b
    if dec.cfg.is_moe:
        assert dec.w["exp_gate"].dtype == ttnn.bfloat4_b, "MoE gate experts must be BFP4"
        assert dec.w["exp_up"].dtype == ttnn.bfloat4_b, "MoE up experts must be BFP4"
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=64, block_size=32)
    assert kv["k"].dtype == ttnn.bfloat8_b and kv["v"].dtype == ttnn.bfloat8_b


# --------------------------------------------------------------------------- #
# Experimental decode QKV fusion: prove the packed [Q|K|V] split itself.
#
# This is intentionally narrower than the end-to-end decode tests below.  It
# catches head-order, logical-batch/padded-batch, and Blackhole interleaved-reader
# regressions in the exact op/layout selected by TT_LAGUNA_FUSE_QKV_DECODE.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("num_heads", [48, 64], ids=["full_attention", "sliding_attention"])
def test_fused_decode_qkv_split_matches_packed_layout(device, num_heads, batch):
    num_kv_heads, head_dim = 8, 128
    q_width = num_heads * head_dim
    kv_width = num_kv_heads * head_dim
    torch.manual_seed(1000 + num_heads + batch)
    packed = torch.randn(1, 1, batch, q_width + 2 * kv_width, dtype=torch.bfloat16)
    packed_tt = ttnn.from_torch(
        packed,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_memcfg = ttnn.create_sharded_memory_config(
        shape=(32, head_dim),
        core_grid=ttnn.CoreGrid(y=4, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    q_tt, k_tt, v_tt = ttnn.experimental.nlp_create_qkv_heads_decode(
        packed_tt,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        overlap_qk_coregrid=True,
        memory_config=output_memcfg,
    )

    expected = (
        packed[..., :q_width].reshape(1, batch, num_heads, head_dim),
        packed[..., q_width : q_width + kv_width].reshape(1, batch, num_kv_heads, head_dim),
        packed[..., q_width + kv_width :].reshape(1, batch, num_kv_heads, head_dim),
    )
    for name, got_tt, want in zip(("q", "k", "v"), (q_tt, k_tt, v_tt), expected, strict=True):
        got = ttnn.to_torch(got_tt)
        assert got.shape == want.shape, f"{name} logical shape {tuple(got.shape)} != {tuple(want.shape)}"
        assert torch.equal(got, want), f"{name} fused split changed packed head values at batch={batch}"


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
# Multi-step sequential decode
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
# Determinism: identical inputs -> bit-identical outputs
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
# Repeated-run stress: many decode steps stay correct + deterministic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_repeated_decode_stress(device, hf_config, layer):
    ctx, raw = _ctx(hf_config, layer)
    S, steps = 40, 24
    max_seq = S + steps + 8
    dec = _decoder(hf_config, layer, device, max_seq)
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=max_seq, block_size=32)
    pt = dec.make_page_table(1, kv["blocks_per_user"])
    torch.manual_seed(21)
    x = torch.randn(1, S, HIDDEN) * 0.5
    _, pkv = R.reference_forward(ctx, x)
    dec.prefill_forward(_tt(x, device), kv, pt, user_id=0, start_pos=0)
    worst = 1.0
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
        xin = _tt(xd.reshape(1, 1, 1, HIDDEN), device)
        out = dec.decode_forward(xin, cur, ridx, pt, kv)
        got = ttnn.to_torch(out).float().reshape(1, 1, HIDDEN)
        worst = min(worst, _pcc(got, ref))
        assert worst >= PCC_BAR, f"layer {layer} stress step {t} pos={pos} PCC {worst:.5f}"


# --------------------------------------------------------------------------- #
# TRACED decode correctness (capture once, replay with in-place updates)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_decode_traced_pcc(device, hf_config, layer):
    ctx, raw = _ctx(hf_config, layer)
    S, steps = 48, 4
    dec = _decoder(hf_config, layer, device)
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=S + steps + 8, block_size=32)
    pt = dec.make_page_table(1, kv["blocks_per_user"])
    torch.manual_seed(123)
    x = torch.randn(1, S, HIDDEN) * 0.5
    _, pkv = R.reference_forward(ctx, x)
    dec.prefill_forward(_tt(x, device), kv, pt, user_id=0, start_pos=0)

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
    dec.decode_forward(x_dev, cur_dev, ridx_dev, pt, kv)
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_dev = dec.decode_forward(x_dev, cur_dev, ridx_dev, pt, kv)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    try:
        for t in range(steps):
            pos = S + t
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
    n_ctx = hf_config.max_position_embeddings - 1
    max_seq = hf_config.max_position_embeddings
    dec = _decoder(hf_config, layer, device, max_seq)
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=max_seq, block_size=32)
    pt = dec.make_page_table(1, kv["blocks_per_user"])
    from transformers.cache_utils import DynamicCache

    cfg = dec.cfg
    g = torch.Generator().manual_seed(99)
    # Varied cached K (query score path runs over all 262144 positions, RoPE at 262143)
    # + ZERO cached V so the output direction == v_new in both bf16 and fp32 → PCC is
    # invariant to the long-context softmax-weighting floor while still exercising full-
    # context cache fill/read, page table over 8192 blocks, and int32 cur_pos + RoPE
    # addressing at the advertised 262144. Cast K/V to the (BFP8) cache dtype first.
    k = (torch.randn(1, cfg.num_kv_heads, n_ctx, cfg.head_dim, generator=g)).to(torch.bfloat16).float()
    v = torch.zeros(1, cfg.num_kv_heads, n_ctx, cfg.head_dim)
    cdt = kv["dtype"]
    ttnn.experimental.paged_fill_cache(kv["k"], ttnn.typecast(_tt(k, device), cdt), pt, batch_idx=0)
    ttnn.experimental.paged_fill_cache(kv["v"], ttnn.typecast(_tt(v, device), cdt), pt, batch_idx=0)
    pkv = DynamicCache(config=ctx.config)
    pkv.update(k.clone(), v.clone(), 0)
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
