# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the production Laguna decoder on a 1×D P150 mesh (D=1, 2, or 4).

Drives ``MultichipDecoder`` (TP/EP=D, replicated residual with per-layer all-reduce for D>1)
on the selected profile and validates every meaningful layer kind against the same
layer-only HF reference at PCC ≥ 0.995 — the same bar/reference as the single-chip optimized suite,
so passing here validates the multichip path against the single-chip TTNN baseline transitively
(the single-chip decoder matches HF ≥ 0.9995; see also ``mc_compare.py`` for the direct
multichip-vs-single-chip-TTNN comparison).

Coverage on the mesh: EP-sharded MoE, local KV-head paged cache, page-table (permuted/nonzero),
tensor current-position, non-aligned logical lengths, batch up to 32, multi-step decode,
determinism, TRACED decode replay (CCL + EP in-trace), forced-chunk / sequence-pipelined long
prefill, and profile maximum-context decode addressing.

Select ``LAGUNA_PROFILE=p150|p150x2|p150x4`` before pytest collection. The default is the
established p150x4 regression profile. Fabric is configured before the mesh opens.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W
from models.autoports.poolside_laguna_xs_2_1.tests.laguna_test_utils import close_mesh, open_mesh, resolve_profile
from models.autoports.poolside_laguna_xs_2_1.tt.multichip_decoder import MultichipDecoder


# The qualification path must match production: packed gate/up is the default. The unpacked path is
# retained only for explicit A/B runs with LAGUNA_PACK_GATE_UP=0 (LAGUNA_MC_CLASS=baseline remains a
# compatibility alias for old runbooks).
_PACK_GATE_UP_REQUESTED = (
    os.environ.get("LAGUNA_PACK_GATE_UP", "1").strip().lower() not in {"0", "false"}
    and os.environ.get("LAGUNA_MC_CLASS") != "baseline"
)


class _DECODER_CLS(MultichipDecoder):
    PACK_GATE_UP = _PACK_GATE_UP_REQUESTED


PROFILE = resolve_profile(trace_region_size=200_000_000)


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
    dev = open_mesh(ttnn, PROFILE)
    yield dev
    close_mesh(ttnn, dev)


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
            raw, hf_config=hf_config, layer_idx=layer, mesh_device=device, max_seq_len=PROFILE.max_context
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
    D = PROFILE.num_devices
    assert dec.D == D and device.get_num_devices() == D
    assert dec.PACK_GATE_UP is _PACK_GATE_UP_REQUESTED
    assert dec.cfg.num_kv_heads == 8 // D
    assert dec.cfg.num_heads == (48 // D if layer in (0, 4) else 64 // D)
    assert dec.use_dram_sharded and dec.policy.kv_cache == ttnn.bfloat8_b
    if dec.cfg.is_moe:
        assert dec.local_experts == 256 // D and dec.global_experts == 256
        # routed-expert weight is BFP4 + EP-sharded (per-device expert dim == 256/D); the optimized
        # class packs gate+up into exp_gate_up, the baseline keeps them separate as exp_gate.
        ekey = "exp_gate_up" if "exp_gate_up" in dec.w else "exp_gate"
        assert dec.w[ekey].dtype == ttnn.bfloat4_b
        assert dec.w[ekey].shape[1] == 256 // D, "experts must be EP-sharded across the selected mesh"
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=64, block_size=32)
    assert kv["k"].shape[1] == 8 // D, "cache holds local KV heads"


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
    n_ctx = PROFILE.max_context - 1
    max_seq = PROFILE.max_context
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
    # Shard the global KV heads across D; each device cache receives 8/D heads.
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


@pytest.mark.skipif(PROFILE.name != "p150x2", reason="focused qualification of the served D2 decode SDPA")
def test_d2_decode_sdpa_pc_direct_traced_boundaries(device, hf_config):
    """Compare the served D2 SDPA program directly with a non-degenerate torch reference.

    The decoder-layer full-context test above deliberately uses an all-zero cached V and compares
    only the residual-dominated layer output.  That is useful as an addressability smoke, but it can
    retain high PCC when SDPA masks a partial chunk incorrectly or contributes almost nothing.  This
    regression instead keeps varied non-zero V throughout a production-block-size cache, puts an
    attention attractor immediately on each tested boundary and a stronger sentinel immediately
    after it, and checks the captured/replayed SDPA output before WO, residual, or MoE can hide it.
    """

    dec = _decoder(hf_config, FULL_MOE, device)
    cfg = dec.cfg
    pc = dec._sdpa_pc_decode
    assert dec._decode_use_sdpa_pc, "D2 qualification must exercise the custom decode SDPA program"
    assert pc.q_chunk_size == 32
    assert pc.k_chunk_size == 64, "k128 is the known-lossy last-partial-chunk configuration"
    assert pc.exp_approx_mode is False
    assert pc.max_cores_per_head_batch == 16

    block_size = 64  # serve_vllm.sh production setting
    max_seq = PROFILE.max_context
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=max_seq, block_size=block_size)
    assert kv["dtype"] == ttnn.bfloat8_b
    assert kv["blocks_per_user"] * block_size == max_seq

    # Exercise both sides of the first two k64/block64 boundaries and the final partial/exact chunks.
    positions = (62, 63, 64, 126, 127, 128, max_seq - 2, max_seq - 1)
    assert len(positions) <= cfg.head_dim
    g = torch.Generator().manual_seed(20260821)
    k = torch.zeros((1, cfg.num_kv_heads, max_seq, cfg.head_dim), dtype=torch.bfloat16)
    v = torch.randn(
        (1, cfg.num_kv_heads, max_seq, cfg.head_dim), dtype=torch.bfloat16, generator=g
    )
    v.mul_(0.25)

    # Each query uses its own coordinate.  Its in-range key has score ~=16; the immediately
    # out-of-range key has score ~=32 and an opposite value, so even one-token mask leakage is loud.
    for dim, pos in enumerate(positions):
        k[0, :, pos, dim] = 16.0
        if pos + 1 < max_seq:
            k[0, :, pos + 1, dim] = 32.0
            v[0, :, pos + 1, :] = -v[0, :, pos, :]

    # Keep the logical order nontrivial: paged_fill writes each logical block through this permutation,
    # and decode must follow the same table.  This is the production block geometry (64), unlike the
    # legacy block-32 decoder tests.
    page_ids = torch.randperm(kv["blocks_per_user"], generator=g, dtype=torch.int64).to(torch.int32)
    assert not torch.equal(page_ids, torch.arange(kv["blocks_per_user"], dtype=torch.int32))
    pt = ttnn.from_torch(
        page_ids.reshape(1, -1),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        mesh_mapper=_mm(device),
    )
    for cache_name, source in (("k", k), ("v", v)):
        source_tt = ttnn.from_torch(
            source,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=_mm(device),
        )
        ttnn.experimental.paged_fill_cache(
            kv[cache_name], dec._cast_fill(source_tt, kv["dtype"]), pt, batch_idx=0
        )

    # Prefix sums make the exact sparse-key reference cheap even at 131K.  For a query coordinate,
    # every unmarked key scores zero, the current-position attractor scores ~=16, and the score-32
    # sentinel is masked.  V remains random/non-zero at every cached position.
    prefix_v = {}
    running = torch.zeros((cfg.num_kv_heads, cfg.head_dim), dtype=torch.float32)
    cursor = 0
    for pos in positions:
        for start in range(cursor, pos + 1, 4096):
            end = min(start + 4096, pos + 1)
            running.add_(v[0, :, start:end, :].float().sum(dim=1))
        prefix_v[pos] = running.clone()
        cursor = pos + 1

    q_amp = float(torch.tensor(cfg.scaling**-1, dtype=torch.bfloat16))

    def host_q(dim):
        q = torch.zeros((1, 1, cfg.num_heads, cfg.head_dim), dtype=torch.bfloat16)
        q[..., dim] = q_amp
        return q

    def reference(dim, pos):
        score = q_amp * 16.0 * cfg.scaling
        attractor_weight = torch.exp(torch.tensor(score, dtype=torch.float32))
        anchor_v = v[0, :, pos, :].float()
        numerator = prefix_v[pos] + (attractor_weight - 1.0) * anchor_v
        denominator = float(pos) + attractor_weight
        per_kv_head = numerator / denominator
        return per_kv_head.repeat_interleave(cfg.num_kv_groups, dim=0).reshape(
            1, 1, cfg.num_heads, cfg.head_dim
        )

    q_dev = ttnn.from_torch(
        host_q(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=_mm(device),
    )
    cur_dev = _int(torch.tensor([positions[0]], dtype=torch.int32), device)
    kwargs = {
        "cur_pos_tensor": cur_dev,
        "page_table_tensor": pt,
        "scale": cfg.scaling,
        "program_config": pc,
        "compute_kernel_config": dec._sdpa_compute,
        "num_kv_heads": cfg.num_kv_heads,
    }
    ttnn.transformer.paged_scaled_dot_product_attention_decode(q_dev, kv["k"], kv["v"], **kwargs)
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_dev = ttnn.transformer.paged_scaled_dot_product_attention_decode(q_dev, kv["k"], kv["v"], **kwargs)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    try:
        for dim, pos in enumerate(positions):
            q_host = ttnn.from_torch(
                host_q(dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=_mm(device)
            )
            pos_host = ttnn.from_torch(
                torch.tensor([pos], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=_mm(device),
            )
            ttnn.copy_host_to_device_tensor(q_host, q_dev)
            ttnn.copy_host_to_device_tensor(pos_host, cur_dev)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

            got = _compose0(out_dev, device).float().reshape(1, 1, cfg.num_heads, cfg.head_dim)
            ref = reference(dim, pos)
            pcc = _pcc(got, ref)
            rmse = torch.sqrt(torch.mean((got - ref) ** 2)).item()
            ref_rms = torch.sqrt(torch.mean(ref**2)).item()
            relative_rmse = rmse / max(ref_rms, 1e-8)
            assert torch.isfinite(got).all(), f"D2 direct SDPA output is not finite at pos={pos}"
            assert pcc >= 0.99, f"D2 direct SDPA pos={pos} PCC {pcc:.5f} < 0.99"
            assert relative_rmse <= 0.15, (
                f"D2 direct SDPA pos={pos} relative RMSE {relative_rmse:.5f} > 0.15 "
                f"(RMSE={rmse:.5f}, reference RMS={ref_rms:.5f})"
            )
    finally:
        ttnn.release_trace(device, tid)
