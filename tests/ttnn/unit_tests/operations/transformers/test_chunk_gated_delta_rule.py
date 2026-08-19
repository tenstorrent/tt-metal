# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Absolute-correctness gate for ttnn.transformer.chunk_gated_delta_rule.

The op computes the gated delta rule chunk-parallel: it tiles the sequence, builds a per-chunk
WY/UT transform (prep phase) and runs a time-sequential, value-parallel scan across chunks. Every
other test of this op compares it against another of its own code paths (the multicast on/off
bit-exactness gate in test_chunk_gated_delta_rule_mcast.py), which cannot catch an error in the
chunked algebra itself: both arms would be wrong together.

So the oracle here is the TOKEN-BY-TOKEN recurrence
``models/experimental/gated_attention_gated_deltanet/torch_functional/delta_rule_ops.py``.

Comparison is PCC: the device accumulates the chunked form in
fp32 through a different association order than a serial fp32 recurrence, so the two differ at
rounding. What bounds the gap is that the op's inputs are bf16 -- the reference is fed the same
bf16-rounded values, upcast, so this measures the op's arithmetic and not input quantization.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.experimental.gated_attention_gated_deltanet.torch_functional.delta_rule_ops import (
    l2_norm,
    recurrent_gated_delta_rule,
)
from tests.ttnn.utils_for_testing import check_with_pcc

CHUNK = 32  # the phased op's supported chunk size (Ct=1); 64 splits the WY matrix (see fused_chunk.py)
REPEATS = 8  # extra multicast runs per shape, to give a non-deterministic race a chance to show

# Measured:
PCC_O = 0.99999
PCC_STATE = 0.99999


def _const_tiles(device, chunk_size=CHUNK):
    """The op's constant tiles (mirrors qwen36 fused_chunk.build_fused_const_tiles).

    Passed explicitly so the op stays stateless and trace-safe; when omitted it builds them itself
    with a host upload, which is illegal under trace capture.
    """
    c = chunk_size
    eye = torch.eye(c, dtype=torch.float32)
    tril = torch.tril(torch.ones(c, c, dtype=torch.float32))
    ones = torch.ones(c, c, dtype=torch.float32)
    ii = torch.arange(32).unsqueeze(1)
    jj = torch.arange(32).unsqueeze(0)
    lo_i, lo_j = ii < 16, jj < 16
    masks = torch.cat(
        [(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], dim=1
    )  # [32, 96]: top-left, bottom-right, bottom-left quadrants

    def _up(t):
        return ttnn.from_torch(t.reshape(1, 1, *t.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    return (_up(eye), _up(tril), _up(ones), _up(masks))


@pytest.mark.skipif(not is_blackhole(), reason="phased chunk_gated_delta_rule is Blackhole-only")
@pytest.mark.parametrize(
    "batch, num_k_heads, num_v_heads",
    [
        (1, 4, 12),  # Qwen3.6-27B per-device shape at TP-4 (GQA group 3)
        (1, 16, 48),  # Qwen3.6-27B single-device shape (GQA group 3)
        (1, 12, 12),  # no GQA: group 1, so the head-map is the identity
        (2, 4, 12),  # batch > 1: BH = 24 independent scans
    ],
    ids=lambda v: str(v),
)
@pytest.mark.parametrize("seq_len", [CHUNK, 128, 256], ids=lambda v: f"T{v}")
@pytest.mark.parametrize("with_initial_state", [False, True], ids=["s0=0", "s0=rand"])
def test_chunk_vs_recurrent_reference(device, batch, num_k_heads, num_v_heads, seq_len, with_initial_state):
    """Chunk-parallel device op vs the token-by-token torch recurrence."""
    torch.manual_seed(20260910)
    B, T, Dk, Dv = batch, seq_len, 128, 128
    G = num_v_heads // num_k_heads
    assert num_v_heads % num_k_heads == 0

    grid = device.compute_with_storage_grid_size()
    if B * num_v_heads > grid.x * grid.y:
        pytest.skip(f"BH={B * num_v_heads} exceeds the {grid.x}x{grid.y} grid (scan needs a core per head)")

    # Inputs in the op's numeric regime: q/k L2-normalized upstream (the GDN layer normalizes over
    # the head dim), beta in (0,1) from a sigmoid, g <= 0 from -softplus.
    q = l2_norm(torch.randn(B, T, num_k_heads, Dk, dtype=torch.float32), dim=-1)
    k = l2_norm(torch.randn(B, T, num_k_heads, Dk, dtype=torch.float32), dim=-1)
    v = torch.randn(B, T, num_v_heads, Dv, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(B, T, num_v_heads, dtype=torch.float32))
    g = -torch.nn.functional.softplus(torch.randn(B, T, num_v_heads, dtype=torch.float32)) * 0.5
    s0 = 0.1 * torch.randn(B, num_v_heads, Dk, Dv, dtype=torch.float32) if with_initial_state else None

    # The op casts q/k/v to bf16 internally, so quantize HERE and give the reference the identical
    # values. Without this the comparison would be dominated by input rounding, not the op.
    q_bf, k_bf, v_bf = (t.to(torch.bfloat16) for t in (q, k, v))

    def dev(t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    eye, tril, ones, masks = _const_tiles(device)
    o_tt, fs_tt = ttnn.transformer.chunk_gated_delta_rule(
        dev(q_bf, ttnn.bfloat16),
        dev(k_bf, ttnn.bfloat16),
        dev(v_bf, ttnn.bfloat16),
        dev(g, ttnn.float32),
        dev(beta, ttnn.float32),
        initial_state=dev(s0, ttnn.float32) if s0 is not None else None,
        output_final_state=True,
        chunk_size=CHUNK,
        eye=eye,
        tril=tril,
        ones=ones,
        masks=masks,
    )
    o_dev = ttnn.to_torch(o_tt).float().reshape(B, T, num_v_heads, Dv)
    fs_dev = ttnn.to_torch(fs_tt).float().reshape(B, num_v_heads, Dk, Dv)

    # Reference: same bf16 values upcast, q/k GQA-expanded to num_v_heads (the op's prep reader maps
    # value-head hv -> key-head hv//G internally), scale applied by the reference's default 1/sqrt(K).
    q_ref = q_bf.float().repeat_interleave(G, dim=2)
    k_ref = k_bf.float().repeat_interleave(G, dim=2)
    o_ref, fs_ref = recurrent_gated_delta_rule(
        q_ref,
        k_ref,
        v_bf.float(),
        beta,
        g,
        initial_state=s0,
        output_final_state=True,
        use_qk_l2norm=False,  # already normalized above, and the op rejects the flag
    )

    ok_o, pcc_o = check_with_pcc(o_ref, o_dev, PCC_O)
    ok_s, pcc_s = check_with_pcc(fs_ref, fs_dev, PCC_STATE)
    # Report both before asserting: when one drifts, the other says whether the scan's carried state
    # or only its per-token read-out is affected.
    print(f"\nPCC o={pcc_o} final_state={pcc_s}")
    assert ok_o, f"o vs recurrent reference: {pcc_o}"
    assert ok_s, f"final_state vs recurrent reference: {pcc_s}"


# --------------------------------------------------------------------------------------------
# Bit-exactness gate for the scan's shared-input multicast.
#
# The scan multicasts its six shared V-independent inputs (kd, q_decay, intra, k_dec_t, dl, t_inv)
# from one sender core per head into the sibling V-block cores' CBs instead of every sibling
# re-reading identical DRAM pages. It forwards the exact bytes the sender read into the same CB
# indices, so the outputs must be BIT-IDENTICAL with it on and off -- any difference is a bug, not
# numerical noise. This is the complement to the PCC test above: that one has an intrinsic ~5e-3
# relative floor (chunk-parallel vs serial fp32 association), so corruption below it is invisible
# there at any tolerance; this comparison's floor is exactly zero.
#
# `use_mcast` is a hashed attribute of ChunkGdnScanParams, so the two calls compile two distinct
# cached scan programs. The program-cache assertion below checks that, which is what keeps the A/B
# non-vacuous: were the argument no longer threaded or hashed, both runs would share one program.
# --------------------------------------------------------------------------------------------


def _scan_nv(device, bh, vt):
    """Replicates distribute_scan's row-aligned NV selection (largest divisor of vt whose 1xNV
    head rectangles fit the padded grid)."""
    grid = device.compute_with_storage_grid_size()
    for cand in range(vt, 0, -1):
        if vt % cand != 0 or cand > grid.x:
            continue
        if bh <= (grid.x // cand) * grid.y:
            return cand
    return 1


def _run_op(device, tensors, const_tiles, initial_state, use_mcast):
    q, k, v, g, beta = tensors
    eye, tril, ones, masks = const_tiles
    o, fs = ttnn.transformer.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=CHUNK,
        eye=eye,
        tril=tril,
        ones=ones,
        masks=masks,
        use_mcast=use_mcast,
    )
    o_t = ttnn.to_torch(o)
    fs_t = ttnn.to_torch(fs)
    ttnn.deallocate(o)
    ttnn.deallocate(fs)
    return o_t, fs_t


@pytest.mark.skipif(not is_blackhole(), reason="phased chunk_gated_delta_rule is Blackhole-only")
@pytest.mark.parametrize(
    "batch, num_k_heads, num_v_heads, want_mcast",
    [
        (1, 4, 12, True),  # TP-4-like per-device shape: BH=12 -> NV=4 on a 13x10 grid, fan-out 3
        (1, 16, 48, True),  # single-device Qwen3.6 shape: BH=48 -> NV=2, fan-out 1
        (2, 16, 48, False),  # batched prefill: BH=96 -> NV=1, multicast degenerates to plain reader
    ],
)
@pytest.mark.parametrize("with_initial_state", [False, True])
def test_scan_mcast_bit_exact(device, monkeypatch, batch, num_k_heads, num_v_heads, want_mcast, with_initial_state):
    torch.manual_seed(20260819)
    B, T, Dk, Dv = batch, 256, 128, 128
    BH = B * num_v_heads

    grid = device.compute_with_storage_grid_size()
    if BH > grid.x * grid.y:
        pytest.skip(f"BH={BH} exceeds the {grid.x}x{grid.y} compute grid (scan needs a core per head)")
    nv = _scan_nv(device, BH, Dv // 32)
    if want_mcast and nv == 1:
        pytest.skip(f"grid {grid.x}x{grid.y} gives NV=1 for BH={BH}: multicast path not exercised")

    # Neutralize ambient GDN debug/profiling knobs that would bypass or fork the scan path.
    monkeypatch.setenv("QWEN_GDN_PHASED", "1")
    monkeypatch.delenv("QWEN_GDN_SCAN_SERIAL", raising=False)
    # QWEN_GDN_DUMP is read once via a function-local static; delenv helps only if the op has not
    # run yet in this process — kept for hygiene.
    monkeypatch.delenv("QWEN_GDN_DUMP", raising=False)

    # Realistic-shaped inputs; bit-exactness holds for any values, but keep them in the op's
    # numeric regime (L2-normalized keys upstream, beta in (0,1), g <= 0).
    q = torch.randn(B, T, num_k_heads, Dk, dtype=torch.bfloat16)
    k = torch.randn(B, T, num_k_heads, Dk, dtype=torch.bfloat16)
    v = torch.randn(B, T, num_v_heads, Dv, dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn(B, T, num_v_heads, dtype=torch.float32))
    g = -torch.nn.functional.softplus(torch.randn(B, T, num_v_heads, dtype=torch.float32)) * 0.5

    def dev(t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tensors = (
        dev(q, ttnn.bfloat16),
        dev(k, ttnn.bfloat16),
        dev(v, ttnn.bfloat16),
        dev(g, ttnn.float32),
        dev(beta, ttnn.float32),
    )
    s0 = None
    if with_initial_state:
        s0_t = 0.1 * torch.randn(B, num_v_heads, Dk, Dv, dtype=torch.float32)
        s0 = dev(s0_t, ttnn.float32)
    const_tiles = _const_tiles(device)

    o_on, fs_on = _run_op(device, tensors, const_tiles, s0, use_mcast=True)
    n_on = device.num_program_cache_entries()

    o_off, fs_off = _run_op(device, tensors, const_tiles, s0, use_mcast=False)
    n_off = device.num_program_cache_entries()

    # The toggle must recompile exactly the scan prim (use_mcast is a hashed attribute); everything
    # else is a cache hit. A delta of 0 means the argument is not threaded to the params or not
    # hashed — and the bit-exact comparison below would be vacuously comparing one program to itself.
    assert n_off - n_on == 1, (
        f"use_mcast toggle compiled {n_off - n_on} new programs (expected exactly the scan prim): "
        "argument not threaded to ChunkGdnScanParams, or not in the program-cache key"
    )

    # Bit-exact: the multicast delivers the same bytes to the same CB slots the plain reader fills.
    assert torch.equal(o_on, o_off), "scan multicast changed o (must be bit-identical)"
    assert torch.equal(fs_on, fs_off), "scan multicast changed final_state (must be bit-identical)"

    # A semaphore race is non-deterministic, so one comparison has little power. The op IS
    # deterministic (verified: 12 identical-config runs are bit-identical), so re-running only the
    # multicast arm against the same plain-reader reference is the cheap way to buy that power —
    # every repeat is a program-cache hit.
    for rep in range(REPEATS):
        o_rep, fs_rep = _run_op(device, tensors, const_tiles, s0, use_mcast=True)
        assert torch.equal(o_on, o_rep), f"multicast o not reproducible on repeat {rep + 1}: race"
        assert torch.equal(fs_on, fs_rep), f"multicast final_state not reproducible on repeat {rep + 1}: race"
