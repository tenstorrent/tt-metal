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

# --------------------------------------------------------------------------------------------
# Shapes under test.
#
# _REGIME_SHAPES exercise the op's internal branches (GQA group 1, batch > 1, V < K).
#
# _QWEN_FAMILY_SHAPES adds every per-chip head geometry at which this op can run for any Qwen model
# and tensor parallelism (TP) setting.
#
# GDN head geometry of the family, from each model's HF config.json:
# `linear_num_key_heads` = 16 and `linear_key_head_dim` = `linear_value_head_dim` = 128 are
# invariant across every member; only `linear_num_value_heads` (HV), and with it the GQA group
# HV/Hk, moves.
#
#   HV  group  models
#   16    1    Qwen3.5-0.8B, Qwen3.5-2B
#   32    2    Qwen3-Next-80B-A3B, Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-35B-A3B, Qwen3.6-35B-A3B
#   48    3    Qwen3.5-27B, Qwen3.6-27B, Qwen3.8-27B
#   64    4    Qwen3.5-122B-A10B, Qwen3.5-397B-A17B
#  128    8    Qwen3.8-2.4T-A95B
#
# Tensor parallelism (TP) distributes both head sets, so (Hk/TP, HV/TP) and the GQA group
# drive the shape of the kernel inputs.
# Hk = 16 caps TP at 16 for every model in the family.
# BH = batch * HV/TP determines the amount of work per core in the scan part. In consequence,
# the test skips a shape whose BH exceeds the grid. On an 11x10 Blackhole chip that is for
# the hv128_tp1 case only (128 heads > 110 cores).
# Note that hv128_tp1 is a rather unrealistic deployment (2.4T model with TP1).
# --------------------------------------------------------------------------------------------
_QWEN_GDN_HK = 16  # linear_num_key_heads, invariant across the family
_QWEN_GDN_DIM = 128  # linear_key_head_dim == linear_value_head_dim, likewise invariant
_QWEN_GDN_HV = (16, 32, 48, 64, 128)  # linear_num_value_heads, per the table above
_QWEN_GDN_TP = (1, 2, 4, 8, 16)  # divisors of Hk = 16

_REGIME_SHAPES = [
    pytest.param(1, 4, 12, 128, 128, id="tp4"),  # Qwen3.6-27B per-device shape at TP-4 (GQA group 3)
    pytest.param(1, 16, 48, 128, 128, id="single_dev"),  # Qwen3.6-27B single-device shape (GQA group 3)
    pytest.param(1, 12, 12, 128, 128, id="no_gqa"),  # no GQA: group 1, so the head-map is the identity
    pytest.param(2, 4, 12, 128, 128, id="batch2"),  # batch > 1: BH = 24 independent scans
    pytest.param(1, 4, 12, 128, 64, id="v64"),  # Small V: V=64
    pytest.param(1, 4, 12, 128, 32, id="v32"),  # Small V: V=32
]

# (Hk, HV) pairs the regime list already runs at batch 1 with K = V = 128; the family sweep skips
# them rather than run the same shape twice under a second id.
_REGIME_HEAD_PAIRS = {(4, 12), (16, 48)}

_QWEN_FAMILY_SHAPES = [
    pytest.param(1, _QWEN_GDN_HK // tp, hv // tp, _QWEN_GDN_DIM, _QWEN_GDN_DIM, id=f"hv{hv}_tp{tp}")
    for hv in _QWEN_GDN_HV
    for tp in _QWEN_GDN_TP
    if (_QWEN_GDN_HK // tp, hv // tp) not in _REGIME_HEAD_PAIRS
]


def test_gated_delta_rule_ops_have_registered_golden_functions():
    assert callable(ttnn.get_golden_function(ttnn.transformer.chunk_gated_delta_rule))
    assert callable(ttnn.get_golden_function(ttnn.transformer.gated_delta_attn_seq))


def test_head_major_layout():
    torch.manual_seed(0)
    B, T, HK, HV, K, V, CS = 2, 4, 2, 6, 2, 3, 2
    q = torch.randn(B, T, HK, K)
    k = torch.randn(B, T, HK, K)
    v = torch.randn(B, T, HV, V)
    g = -torch.rand(B, T, HV)
    beta = torch.sigmoid(torch.randn(B, T, HV))
    golden = ttnn.get_golden_function(ttnn.transformer.chunk_gated_delta_rule)

    token_major, state = golden(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=CS,
        output_final_state=True,
    )
    head_major, hm_state = golden(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=CS,
        output_final_state=True,
        output_head_major=True,
    )

    assert token_major.shape == (B, T, HV, V) and head_major.shape == (B * HV, T, V)
    # Built independently of the implementation's permute/reshape.
    expected = torch.stack([token_major[b, :, h] for b in range(B) for h in range(HV)])
    torch.testing.assert_close(head_major, expected)
    torch.testing.assert_close(hm_state, state)


def test_gqa_expansion_is_repeat_interleave():
    torch.manual_seed(0)
    B, T, HK, HV, K, V, CS = 2, 4, 2, 6, 2, 3, 2
    q = torch.randn(B, T, HK, K)
    k = torch.randn(B, T, HK, K)
    v = torch.randn(B, T, HV, V)
    g = -torch.rand(B, T, HV)
    beta = torch.sigmoid(torch.randn(B, T, HV))
    golden = ttnn.get_golden_function(ttnn.transformer.chunk_gated_delta_rule)

    got, got_state = golden(q, k, v, g, beta, chunk_size=CS, output_final_state=True)
    qe = q.repeat_interleave(HV // HK, dim=2)
    ke = k.repeat_interleave(HV // HK, dim=2)
    want, want_state = golden(qe, ke, v, g, beta, chunk_size=CS, output_final_state=True)
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(got_state, want_state)


def test_gated_delta_attn_seq_golden_matches_documented_scan():
    torch.manual_seed(1)
    batch_heads, num_chunks, chunk_size, key_dim, value_dim = 1, 2, 4, 3, 2
    strict_lower = torch.tril(torch.randn(batch_heads, num_chunks, chunk_size, chunk_size), diagonal=-1)
    L_unit = strict_lower + torch.eye(chunk_size).reshape(1, 1, chunk_size, chunk_size)
    v_beta_sc = torch.randn(batch_heads, num_chunks, chunk_size, value_dim)
    k_bd_sc = torch.randn(batch_heads, num_chunks, chunk_size, key_dim)
    intra_attn = torch.randn(batch_heads, num_chunks, chunk_size, chunk_size)
    q_decay = torch.randn(batch_heads, num_chunks, chunk_size, key_dim)
    k_decay_t = torch.randn(batch_heads, num_chunks, key_dim, chunk_size)
    dl_exp = torch.rand(batch_heads, num_chunks, 1, 1)
    L_inv = torch.empty(batch_heads, num_chunks, chunk_size, 32)
    initial_state = torch.randn(batch_heads, key_dim, value_dim)

    expected_outputs = []
    expected_state = initial_state.clone()
    for chunk in range(num_chunks):
        v_cor = torch.linalg.solve_triangular(L_unit[:, chunk], v_beta_sc[:, chunk], upper=False, unitriangular=False)
        k_cum = torch.linalg.solve_triangular(L_unit[:, chunk], k_bd_sc[:, chunk], upper=False, unitriangular=False)
        v_new = v_cor - k_cum @ expected_state
        expected_outputs.append(q_decay[:, chunk] @ expected_state + intra_attn[:, chunk] @ v_new)
        expected_state = expected_state * dl_exp[:, chunk] + k_decay_t[:, chunk] @ v_new

    golden = ttnn.get_golden_function(ttnn.transformer.gated_delta_attn_seq)
    actual_output, actual_state = golden(
        L_unit,
        v_beta_sc,
        k_bd_sc,
        intra_attn,
        q_decay,
        k_decay_t,
        dl_exp,
        L_inv,
        initial_state=initial_state,
    )

    torch.testing.assert_close(actual_output, torch.stack(expected_outputs, dim=1))
    torch.testing.assert_close(actual_state, expected_state)


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
@pytest.mark.parametrize("batch, num_k_heads, num_v_heads, key_dim, val_dim", _REGIME_SHAPES + _QWEN_FAMILY_SHAPES)
@pytest.mark.parametrize("seq_len", [CHUNK, 128, 256], ids=lambda v: f"T{v}")
@pytest.mark.parametrize("with_initial_state", [False, True], ids=["s0=0", "s0=rand"])
def test_chunk_vs_recurrent_reference(
    device, batch, num_k_heads, num_v_heads, key_dim, val_dim, seq_len, with_initial_state
):
    """Chunk-parallel device op vs the token-by-token torch recurrence."""
    torch.manual_seed(20260910)
    B, T, Dk, Dv = batch, seq_len, key_dim, val_dim
    G = num_v_heads // num_k_heads
    assert (
        num_v_heads % num_k_heads == 0
    ), f"num_v_heads ({num_v_heads}) must be a multiple of num_k_heads ({num_k_heads}) for the GQA head-map"

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
# ChunkGdnPhasedProgramConfig.use_mcast lands as a hashed attribute, so the two calls compile two
# distinct cached scan programs. The program-cache assertion below checks that, which is what keeps
# the A/B comparison meaningful.
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


def _run_op(device, tensors, const_tiles, initial_state, chunk_size, use_mcast):
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
        chunk_size=chunk_size,
        eye=eye,
        tril=tril,
        ones=ones,
        masks=masks,
        program_config=ttnn.ChunkGdnPhasedProgramConfig(use_mcast=use_mcast),
    )
    o_t = ttnn.to_torch(o)
    fs_t = ttnn.to_torch(fs)
    ttnn.deallocate(o)
    ttnn.deallocate(fs)
    return o_t, fs_t


# NV (v-blocks per head) is grid-dependent, so the comments below give it for the 11x10 Blackhole
# worker grid this was developed on; the test recomputes it and skips rather than assuming.
# The last three rows are the branches the shared-input transfer counts actually differ on --
# K != V changes the ck/kc counts, chunk_size=64 makes Ct=2, and T == chunk_size makes NC==1 a
# single-chunk handshake with no steady state. All three are supported and verified bit-exact.
@pytest.mark.skipif(not is_blackhole(), reason="phased chunk_gated_delta_rule is Blackhole-only")
@pytest.mark.parametrize(
    "batch, num_k_heads, num_v_heads, key_dim, val_dim, seq_len, chunk, want_mcast",
    [
        (1, 4, 12, 128, 128, 256, 32, True),  # TP-4 per-device shape: BH=12 -> NV=4, fan-out 3
        (1, 16, 48, 128, 128, 256, 32, True),  # single-device Qwen3.6 shape: BH=48 -> NV=2, fan-out 1
        (2, 16, 48, 128, 128, 256, 32, False),  # batched prefill: BH=96 -> NV=1, degenerates to plain reader
        (1, 4, 12, 64, 128, 256, 32, True),  # K != V: kd/q_decay/k_dec_t shrink, v_beta does not
        (1, 4, 12, 128, 128, 256, 64, True),  # chunk_size=64 -> Ct=2: two tile-rows per chunk
        (1, 4, 12, 128, 128, 32, 32, True),  # T == chunk_size -> NC==1: single-chunk handshake
        (1, 4, 12, 128, 64, 256, 32, True),  # small V: Ct*Vt < 3, the prep mask-slot capacity regime
    ],
    ids=[
        "tp4",
        "single_dev",
        "nv1_plain",
        "k_ne_v",
        "chunk64_ct2",
        "nc1",
        "small_v",
    ],
)
@pytest.mark.parametrize("with_initial_state", [False, True])
def test_scan_mcast_bit_exact(
    device,
    batch,
    num_k_heads,
    num_v_heads,
    key_dim,
    val_dim,
    seq_len,
    chunk,
    want_mcast,
    with_initial_state,
):
    torch.manual_seed(20260819)
    B, T, Dk, Dv = batch, seq_len, key_dim, val_dim
    BH = B * num_v_heads

    grid = device.compute_with_storage_grid_size()
    if BH > grid.x * grid.y:
        pytest.skip(f"BH={BH} exceeds the {grid.x}x{grid.y} compute grid (scan needs a core per head)")
    nv = _scan_nv(device, BH, Dv // 32)
    if want_mcast and nv == 1:
        pytest.skip(f"grid {grid.x}x{grid.y} gives NV=1 for BH={BH}: multicast path not exercised")

    # Realistic-shaped inputs; bit-exactness holds for any values, but keep them in the op's
    # numeric regime (L2-normalized q/k upstream, beta in (0,1), g <= 0).
    q = l2_norm(torch.randn(B, T, num_k_heads, Dk, dtype=torch.float32), dim=-1).to(torch.bfloat16)
    k = l2_norm(torch.randn(B, T, num_k_heads, Dk, dtype=torch.float32), dim=-1).to(torch.bfloat16)
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
    const_tiles = _const_tiles(device, chunk)

    o_on, fs_on = _run_op(device, tensors, const_tiles, s0, chunk, use_mcast=True)
    n_on = device.num_program_cache_entries()

    o_off, fs_off = _run_op(device, tensors, const_tiles, s0, chunk, use_mcast=False)
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
        o_rep, fs_rep = _run_op(device, tensors, const_tiles, s0, chunk, use_mcast=True)
        assert torch.equal(o_on, o_rep), f"multicast o not reproducible on repeat {rep + 1}: race"
        assert torch.equal(fs_on, fs_rep), f"multicast final_state not reproducible on repeat {rep + 1}: race"


# The kernels run one arithmetic (HiFi4, fp32 destination accumulation, no approx) on every path;
# compute_kernel_config may spell it out but may not change it. Knobs these kernels do not use
# (packer_l1_acc) are accepted and ignored.
_UNSUPPORTED_COMPUTE_CONFIGS = {
    "HiFi2": dict(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False),
    "fp16-dest": dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False),
    "approx": dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=True),
}


def _small_inputs(device):
    torch.manual_seed(20260925)
    B, T, Hk, Hv, D = 1, CHUNK, 2, 4, 128

    def dev(t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    q = dev(l2_norm(torch.randn(B, T, Hk, D), dim=-1).to(torch.bfloat16), ttnn.bfloat16)
    k = dev(l2_norm(torch.randn(B, T, Hk, D), dim=-1).to(torch.bfloat16), ttnn.bfloat16)
    v = dev(torch.randn(B, T, Hv, D).to(torch.bfloat16), ttnn.bfloat16)
    g = dev(-torch.nn.functional.softplus(torch.randn(B, T, Hv)) * 0.5, ttnn.float32)
    beta = dev(torch.sigmoid(torch.randn(B, T, Hv)), ttnn.float32)
    return (q, k, v, g, beta), _const_tiles(device)


def _run_with_compute_config(device, tensors, const_tiles, program_config, compute_kernel_config):
    q, k, v, g, beta = tensors
    eye, tril, ones, masks = const_tiles
    o, fs = ttnn.transformer.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        output_final_state=True,
        chunk_size=CHUNK,
        eye=eye,
        tril=tril,
        ones=ones,
        masks=masks,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    o_t, fs_t = ttnn.to_torch(o), ttnn.to_torch(fs)
    ttnn.deallocate(o)
    ttnn.deallocate(fs)
    return o_t, fs_t


@pytest.mark.skipif(not is_blackhole(), reason="chunk_gated_delta_rule is Blackhole-only")
@pytest.mark.parametrize(
    "make_program_config",
    [ttnn.ChunkGdnMonoProgramConfig, ttnn.ChunkGdnPhasedProgramConfig, ttnn.ChunkGdnFusedProgramConfig],
    ids=["mono", "phased", "fused"],
)
def test_compute_kernel_config_contract(device, expect_error, make_program_config):
    """An explicit config equal to the supported arithmetic (plus an unused knob) is bit-identical to
    passing none; any other fidelity / accumulation / approx setting is rejected on every path."""
    tensors, const_tiles = _small_inputs(device)
    pc = make_program_config()
    o_ref, fs_ref = _run_with_compute_config(device, tensors, const_tiles, pc, None)

    explicit = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
    )
    o, fs = _run_with_compute_config(device, tensors, const_tiles, pc, explicit)
    assert torch.equal(o, o_ref) and torch.equal(fs, fs_ref), "spelling out the default arithmetic changed the result"

    for name, kwargs in _UNSUPPORTED_COMPUTE_CONFIGS.items():
        with expect_error(RuntimeError, "HiFi4 with fp32 destination accumulation"):
            _run_with_compute_config(device, tensors, const_tiles, pc, ttnn.WormholeComputeKernelConfig(**kwargs))
            pytest.fail(f"{name} was accepted; the op must reject arithmetic other than the one it was validated at")


# --------------------------------------------------------------------------------------------
# State-decay precision gate.
#
# The regime tests above draw g from -softplus(randn)/2 and compare against a float32 recurrence,
# so per-chunk decay factors are ~1e-5 and the carried state is almost entirely the latest chunk's
# update: an error in the decay path (S <- dl*S + k_dec_t @ v_new) is invisible there. This test
# uses g = -0.02*|N(0,1)| (dl ~ 0.6 per 32-token chunk, so every chunk's state survives for many
# chunks) and a float64 token-by-token reference, and bounds the rms error of the final state.
# Calibrated on QB2 (p300) at this shape/seed: 4.5e-4 with the scan's o and state updates folded
# into DST accumulation (the shipped form); the earlier kernel that packed and re-read both partial
# sums measured 5.05e-4. The bound has ~10 % headroom over the shipped form.
# --------------------------------------------------------------------------------------------


def _fp64_token_recurrence(q, k, v, beta, g, s0, scale):
    """Token-by-token gated delta rule in float64 (the same semantics as recurrent_gated_delta_rule,
    which computes in float32); q/k/v/beta/g are [B, T, H, ...] head-expanded, s0 is [B, H, K, V]."""
    q, k, v = (t.double().transpose(1, 2) for t in (q, k, v))  # [B, H, T, D]
    beta, g = beta.double().transpose(1, 2), g.double().transpose(1, 2)  # [B, H, T]
    h = s0.double().clone()
    o = torch.zeros(*v.shape, dtype=torch.float64)
    for i in range(q.shape[2]):
        h = h * g[:, :, i].exp()[..., None, None]
        b_v = v[:, :, i] - (h * k[:, :, i][..., None]).sum(-2)
        b_v = b_v * beta[:, :, i][..., None]
        h = h + k[:, :, i].unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = ((q[:, :, i] * scale)[..., None] * h).sum(-2)
    return o.transpose(1, 2), h


@pytest.mark.skipif(not is_blackhole(), reason="chunk_gated_delta_rule is Blackhole-only")
def test_state_decay_vs_fp64_reference(device):
    torch.manual_seed(20260925)
    B, T, Hk, Hv, Dk, Dv = 1, 2048, 4, 12, 128, 128  # the 27B TP-4 slice: BH=12, G=3, NC=64
    G = Hv // Hk
    grid = device.compute_with_storage_grid_size()
    if B * Hv > grid.x * grid.y:
        pytest.skip(f"BH={B * Hv} exceeds the {grid.x}x{grid.y} grid")

    q = l2_norm(torch.randn(B, T, Hk, Dk), dim=-1).to(torch.bfloat16)
    k = l2_norm(torch.randn(B, T, Hk, Dk), dim=-1).to(torch.bfloat16)
    v = (0.5 * torch.randn(B, T, Hv, Dv)).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(B, T, Hv))
    g = -0.02 * torch.randn(B, T, Hv).abs()
    s0 = 0.05 * torch.randn(B, Hv, Dk, Dv)

    def dev(t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    eye, tril, ones, masks = _const_tiles(device)
    o_tt, fs_tt = ttnn.transformer.chunk_gated_delta_rule(
        dev(q, ttnn.bfloat16),
        dev(k, ttnn.bfloat16),
        dev(v, ttnn.bfloat16),
        dev(g, ttnn.float32),
        dev(beta, ttnn.float32),
        initial_state=dev(s0, ttnn.float32),
        output_final_state=True,
        chunk_size=CHUNK,
        eye=eye,
        tril=tril,
        ones=ones,
        masks=masks,
    )
    o_dev = ttnn.to_torch(o_tt).double().reshape(B, T, Hv, Dv)
    fs_dev = ttnn.to_torch(fs_tt).double().reshape(B, Hv, Dk, Dv)

    q_ref = q.float().repeat_interleave(G, dim=2)
    k_ref = k.float().repeat_interleave(G, dim=2)
    o_ref, fs_ref = _fp64_token_recurrence(q_ref, k_ref, v.float(), beta, g, s0, scale=Dk**-0.5)

    ok_o, pcc_o = check_with_pcc(o_ref.float(), o_dev.float(), 0.9999)
    ok_s, pcc_s = check_with_pcc(fs_ref.float(), fs_dev.float(), PCC_STATE)
    rms = (fs_dev - fs_ref).pow(2).mean().sqrt().item()
    print(f"\nPCC o={pcc_o} final_state={pcc_s}; final-state rms error vs fp64 = {rms:.3e}")
    assert ok_o, f"o vs fp64 recurrence: {pcc_o}"
    assert ok_s, f"final_state vs fp64 recurrence: {pcc_s}"
    assert rms <= 5.0e-4, f"final-state rms error {rms:.3e} > 5.0e-4 (shipped kernel: 4.5e-4)"
