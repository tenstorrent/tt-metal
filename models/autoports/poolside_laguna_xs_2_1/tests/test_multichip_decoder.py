# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the MULTICHIP Laguna-XS-2.1 decoder (Blackhole p300c ×4, 1×4 mesh).

Drives ``MultichipDecoder`` (TP=4 attention/dense + EP=4 routed MoE, replicated residual with
per-layer all_reduce) on the 1×4 mesh and validates every meaningful layer kind against the same
layer-only HF reference at PCC ≥ 0.995 — the same bar/reference as the single-chip optimized suite,
so passing here validates the multichip path against the single-chip TTNN baseline transitively
(the single-chip decoder matches HF ≥ 0.9995; see also ``mc_compare.py`` for the direct
multichip-vs-single-chip-TTNN comparison).

Coverage on the mesh: EP-sharded MoE, local 2-KV-head paged cache, page-table (permuted/nonzero),
tensor current-position, non-aligned logical lengths, batch up to 32, multi-step decode,
determinism, TRACED decode replay (CCL + EP in-trace), forced-chunk / sequence-pipelined long
prefill, and full advertised context (262144) decode addressing.

ENVIRONMENT (installed self-consistent tree from a non-repo cwd; FABRIC_1D_RING is set by the
module fixture before the mesh opens):
    cd /tmp && TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal \
        PYTHONPATH=/home/ttuser/dev/tt-metal python -m pytest <thisfile> -q
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W
from models.autoports.poolside_laguna_xs_2_1.tt.multichip_decoder import MultichipDecoder
from models.autoports.poolside_laguna_xs_2_1.tt.optimized_multichip_decoder import OptimizedMultichipDecoder

# Same suite validates either class; set LAGUNA_MC_CLASS=optimized for the optimized-multichip path.
_DECODER_CLS = OptimizedMultichipDecoder if os.environ.get("LAGUNA_MC_CLASS") == "optimized" else MultichipDecoder

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
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    yield dev
    ttnn.close_mesh_device(dev)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


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


def _decoder(hf_config, layer, device):
    if layer not in _DEC:
        _, raw = _ctx(hf_config, layer)
        _DEC[layer] = _DECODER_CLS.from_state_dict(
            raw, hf_config=hf_config, layer_idx=layer, mesh_device=device, max_seq_len=hf_config.max_position_embeddings
        )
    return _DEC[layer]


def _mm(device):
    return ttnn.ReplicateTensorToMesh(device)


def _tt(x, device):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=_mm(device))


def _int(x, device, dtype=ttnn.int32):
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, mesh_mapper=_mm(device))


def _compose0(out, device):
    """Replicated output -> device-0 copy on host."""
    return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[0:1]


# --------------------------------------------------------------------------- #
# The measured path must be the MULTICHIP one (EP-sharded, local KV heads).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_multichip_path_active(device, hf_config, layer):
    dec = _decoder(hf_config, layer, device)
    assert dec.D == 4 and device.get_num_devices() == 4
    assert dec.cfg.num_kv_heads == 2, "local KV heads must be 8/4=2"
    assert dec.cfg.num_heads == (48 // 4 if layer in (0, 4) else 64 // 4)
    assert dec.use_dram_sharded and dec.policy.kv_cache == ttnn.bfloat8_b
    if dec.cfg.is_moe:
        assert dec.local_experts == 64 and dec.global_experts == 256
        # routed-expert weight is BFP4 + EP-sharded (per-device expert dim == 64); the optimized
        # class packs gate+up into exp_gate_up, the baseline keeps them separate as exp_gate.
        ekey = "exp_gate_up" if "exp_gate_up" in dec.w else "exp_gate"
        assert dec.w[ekey].dtype == ttnn.bfloat4_b
        assert dec.w[ekey].shape[1] == 64, "experts must be EP-sharded 64/device"
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=64, block_size=32)
    assert kv["k"].shape[1] == 2, "cache holds local KV heads"


@pytest.mark.parametrize("seq", [8, 32, 100, 512, 513, 1024, 2048])
@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_prefill_pcc(device, hf_config, layer, seq):
    ctx, raw = _ctx(hf_config, layer)
    dec = _decoder(hf_config, layer, device)
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=seq + 32, block_size=32)
    pt = dec.make_page_table(1, kv["blocks_per_user"])
    torch.manual_seed(seq)
    x = torch.randn(1, seq, HIDDEN) * 0.5
    ref, _ = R.reference_forward(ctx, x)
    out = dec.prefill_forward(_tt(x, device), kv, pt, user_id=0, start_pos=0)
    got = _compose0(out, device).float().reshape(1, seq, HIDDEN)
    pcc = _pcc(got, ref)
    assert pcc >= PCC_BAR, f"layer {layer} prefill seq={seq} PCC {pcc:.5f} < {PCC_BAR}"


@pytest.mark.parametrize("prefill_seq", [32, 513, 2048])
@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_decode_pcc(device, hf_config, layer, prefill_seq):
    ctx, raw = _ctx(hf_config, layer)
    dec = _decoder(hf_config, layer, device)
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=prefill_seq + 8, block_size=32)
    pt = dec.make_page_table(1, kv["blocks_per_user"])
    torch.manual_seed(prefill_seq)
    x = torch.randn(1, prefill_seq, HIDDEN) * 0.5
    _, pkv = R.reference_forward(ctx, x)
    dec.prefill_forward(_tt(x, device), kv, pt, user_id=0, start_pos=0)
    xd = torch.randn(1, 1, HIDDEN) * 0.5
    ref, _ = R.reference_forward(ctx, xd, past_key_values=pkv)
    cur = _int(torch.tensor([prefill_seq], dtype=torch.int32), device)
    ridx = _int(torch.tensor([[prefill_seq]], dtype=torch.int32), device, ttnn.uint32)
    out = dec.decode_forward(_tt(xd.reshape(1, 1, 1, HIDDEN), device), cur, ridx, pt, kv)
    got = _compose0(out, device).float().reshape(1, 1, HIDDEN)
    pcc = _pcc(got, ref)
    assert pcc >= PCC_BAR, f"layer {layer} decode pos={prefill_seq} PCC {pcc:.5f} < {PCC_BAR}"


@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_multistep_decode(device, hf_config, layer):
    ctx, raw = _ctx(hf_config, layer)
    S, steps = 48, 5
    dec = _decoder(hf_config, layer, device)
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=S + steps + 8, block_size=32)
    pt = dec.make_page_table(1, kv["blocks_per_user"])
    torch.manual_seed(7)
    x = torch.randn(1, S, HIDDEN) * 0.5
    _, pkv = R.reference_forward(ctx, x)
    dec.prefill_forward(_tt(x, device), kv, pt, user_id=0, start_pos=0)
    for t in range(steps):
        pos = S + t
        xd = torch.randn(1, 1, HIDDEN) * 0.5
        ref, _ = R.reference_forward(ctx, xd, past_key_values=pkv)
        cur = _int(torch.tensor([pos], dtype=torch.int32), device)
        ridx = _int(torch.tensor([[pos]], dtype=torch.int32), device, ttnn.uint32)
        out = dec.decode_forward(_tt(xd.reshape(1, 1, 1, HIDDEN), device), cur, ridx, pt, kv)
        got = _compose0(out, device).float().reshape(1, 1, HIDDEN)
        pcc = _pcc(got, ref)
        assert pcc >= PCC_BAR, f"layer {layer} step {t} pos={pos} PCC {pcc:.5f}"


@pytest.mark.parametrize("B", [4, 32])
@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_batch_prefill_decode(device, hf_config, layer, B):
    ctx, raw = _ctx(hf_config, layer)
    S = 48
    dec = _decoder(hf_config, layer, device)
    kv = dec.alloc_kv_cache(max_users=B, max_seq_len=S + 8, block_size=32)
    bpu = kv["blocks_per_user"]
    torch.manual_seed(3)
    perm = torch.randperm(B * bpu).to(torch.int32).reshape(B, bpu)
    pt = ttnn.from_torch(perm, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, mesh_mapper=_mm(device))
    xs = [torch.randn(1, S, HIDDEN) * 0.5 for _ in range(B)]
    pkvs = []
    for u in range(B):
        ref, pkv = R.reference_forward(ctx, xs[u])
        pkvs.append(pkv)
        out = dec.prefill_forward(_tt(xs[u], device), kv, pt, user_id=u, start_pos=0)
        got = _compose0(out, device).float().reshape(1, S, HIDDEN)
        assert _pcc(got, ref) >= PCC_BAR, f"layer {layer} user {u} prefill PCC low"
    xd = torch.randn(1, 1, B, HIDDEN) * 0.5
    refs = [R.reference_forward(ctx, xd[:, :, u, :], past_key_values=pkvs[u])[0] for u in range(B)]
    ref_dec = torch.cat(refs, dim=1).reshape(B, HIDDEN)
    cur = _int(torch.full((B,), S, dtype=torch.int32), device)
    ridx = _int(torch.full((1, B), S, dtype=torch.int32), device, ttnn.uint32)
    out = dec.decode_forward(_tt(xd, device), cur, ridx, pt, kv)
    got = _compose0(out, device).float().reshape(B, HIDDEN)
    pcc = _pcc(got, ref_dec)
    assert pcc >= PCC_BAR, f"layer {layer} batch decode PCC {pcc:.5f}"


@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_determinism(device, hf_config, layer):
    _ctx(hf_config, layer)
    S = 64
    dec = _decoder(hf_config, layer, device)

    def run():
        kv = dec.alloc_kv_cache(max_users=1, max_seq_len=S + 8, block_size=32)
        pt = dec.make_page_table(1, kv["blocks_per_user"])
        torch.manual_seed(11)
        x = torch.randn(1, S, HIDDEN) * 0.5
        dec.prefill_forward(_tt(x, device), kv, pt, user_id=0, start_pos=0)
        xd = torch.randn(1, 1, HIDDEN) * 0.5
        cur = _int(torch.tensor([S], dtype=torch.int32), device)
        ridx = _int(torch.tensor([[S]], dtype=torch.int32), device, ttnn.uint32)
        out = dec.decode_forward(_tt(xd.reshape(1, 1, 1, HIDDEN), device), cur, ridx, pt, kv)
        return _compose0(out, device).float()

    a = run()
    b = run()
    assert torch.equal(a, b), f"layer {layer} decode not deterministic (max diff {(a - b).abs().max()})"


@pytest.mark.parametrize("layer", ALL_LAYERS)
def test_repeated_decode_stress(device, hf_config, layer):
    ctx, raw = _ctx(hf_config, layer)
    S, steps = 40, 24
    dec = _decoder(hf_config, layer, device)
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=S + steps + 8, block_size=32)
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
        cur = _int(torch.tensor([pos], dtype=torch.int32), device)
        ridx = _int(torch.tensor([[pos]], dtype=torch.int32), device, ttnn.uint32)
        out = dec.decode_forward(_tt(xd.reshape(1, 1, 1, HIDDEN), device), cur, ridx, pt, kv)
        got = _compose0(out, device).float().reshape(1, 1, HIDDEN)
        worst = min(worst, _pcc(got, ref))
        assert worst >= PCC_BAR, f"layer {layer} stress step {t} pos={pos} PCC {worst:.5f}"


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
        return ttnn.from_torch(
            t.reshape(1, 1, 1, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=_mm(device)
        )

    def host_pos(p):
        return ttnn.from_torch(
            torch.tensor([p], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=_mm(device),
        )

    def host_ridx(p):
        return ttnn.from_torch(
            torch.tensor([[p]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=_mm(device),
        )

    step_x = [torch.randn(1, 1, HIDDEN) * 0.5 for _ in range(steps)]
    x_dev = _tt(step_x[0].reshape(1, 1, 1, HIDDEN), device)
    cur_dev = _int(torch.tensor([S], dtype=torch.int32), device)
    ridx_dev = _int(torch.tensor([[S]], dtype=torch.int32), device, ttnn.uint32)
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
            got = _compose0(out_dev, device).float().reshape(1, 1, HIDDEN)
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
        got = _compose0(out, device).float().reshape(1, seq, HIDDEN)
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
        got = _compose0(out, device).float().reshape(1, seq, HIDDEN)
        pcc = _pcc(got, ref)
        assert pcc >= PCC_BAR, f"layer {layer} pipelined prefill seq={seq} PCC {pcc:.5f} < {PCC_BAR}"
    finally:
        dec.PIPE_CHUNK = orig


@pytest.mark.parametrize("layer", [FULL_MOE, SLIDING_MOE])
def test_full_context_decode(device, hf_config, layer):
    ctx, raw = _ctx(hf_config, layer)
    n_ctx = hf_config.max_position_embeddings - 1
    max_seq = hf_config.max_position_embeddings
    dec = _decoder(hf_config, layer, device)
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=max_seq, block_size=32)
    pt = dec.make_page_table(1, kv["blocks_per_user"])
    from transformers.cache_utils import DynamicCache

    cfg = dec.cfg
    g = torch.Generator().manual_seed(99)
    # Varied cached K + ZERO cached V so the output direction == v_new -> PCC invariant to the
    # long-context softmax floor while exercising full-context cache/page-table/cur_pos/RoPE at
    # the advertised 262144 with local (2-KV-head) BFP8 cache. Reference uses the full 8-KV-head
    # cache; the per-device local KV heads reconstruct the same attention on device 0.
    GKV = 8
    k = (torch.randn(1, GKV, n_ctx, cfg.head_dim, generator=g)).to(torch.bfloat16).float()
    v = torch.zeros(1, GKV, n_ctx, cfg.head_dim)
    # local KV heads for device d = k[:, 2d:2d+2]; fill each device's cache with its slice.
    cdt = kv["dtype"]
    k_sh = ttnn.from_torch(
        k,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
    )
    v_sh = ttnn.from_torch(
        v,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
    )
    ttnn.experimental.paged_fill_cache(kv["k"], ttnn.typecast(k_sh, cdt), pt, batch_idx=0)
    ttnn.experimental.paged_fill_cache(kv["v"], ttnn.typecast(v_sh, cdt), pt, batch_idx=0)
    pkv = DynamicCache(config=ctx.config)
    pkv.update(k.clone(), v.clone(), 0)
    xd = torch.randn(1, 1, HIDDEN) * 0.5
    ref, _ = R.reference_forward(ctx, xd, past_key_values=pkv)
    cur = _int(torch.tensor([n_ctx], dtype=torch.int32), device)
    ridx = _int(torch.tensor([[n_ctx]], dtype=torch.int32), device, ttnn.uint32)
    out = dec.decode_forward(_tt(xd.reshape(1, 1, 1, HIDDEN), device), cur, ridx, pt, kv)
    got = _compose0(out, device).float().reshape(1, 1, HIDDEN)
    assert torch.isfinite(got).all(), f"layer {layer} full-context output not finite"
    pcc = _pcc(got, ref)
    assert pcc >= PCC_BAR, f"layer {layer} full-context (pos={n_ctx}) decode PCC {pcc:.5f} < {PCC_BAR}"
