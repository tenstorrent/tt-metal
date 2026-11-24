# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Device unit tests for the GLM-5.3-Flash KDA (Kimi Delta Attention) block.

This is the module carrying the Blackhole -> Wormhole port, so the tests are ordered from the
op level up:

1. ``test_conv_with_carry_matches_plain_conv`` -- the replacement op the port introduced
   (``qkv_causal_conv1d_silu_with_carry``) must compute exactly what the old
   ``chronological_selections`` + ``qkv_causal_conv1d_silu`` pair did on a single chip. This is the
   regression test for the port itself.
2. ``test_kda_prefill_matches_reference`` -- the whole block against a fp32 torch reference of the
   documented math, over one chunk and across two chunks (which exercises the conv-history carry
   that the port changed).
3. ``test_kda_decode_matches_reference`` -- the token-at-a-time recurrence.
4. ``test_prefill_then_decode_agrees`` -- the state written by a prefill chunk must be the state a
   token-by-token decode would have reached; this is what a carry bug breaks first.

The reference implements the formula in ``tt/kda.py``'s docstring and HF
``Glm5NextTextLinearAttention``:

    u          = [x Wq | x Wk | x Wv]
    q,k,v      = SiLU(causal depthwise conv_4(u))       (3-token pre-conv history)
    g          = lower_bound * sigmoid(exp(A_log[h]) * (f_b(f_a(x)) + dt_bias))   per head AND channel
    beta       = sigmoid(x Wb)
    S          = diag(exp g_t) S ; S += k_t (beta_t (v_t - S^T k_t))^T
    o_t        = S^T q_t / sqrt(d)                      (q, k l2-normalized)
    out        = o_proj( RMSNorm(o) * w * sigmoid(g_b(g_a(x))) )
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.experimental.glm53_flash.tt.kda import CONV_K, KDA, KDAStateCache
from models.experimental.glm53_flash.tt.precision import KDA_STATE_DTYPE

pytestmark = pytest.mark.use_module_device({"l1_small_size": 24576})

DRAM = ttnn.DRAM_MEMORY_CONFIG

# Small but structurally faithful: the block is parameterised by H, K, hidden, so a small instance
# exercises the same code paths as the real 64 x 128 x 4096 without a multi-GB weight upload.
H = 4  # heads
K = 64  # head dim
HIDDEN = 256
WIDTH = H * K  # 256, = q|k|v block width


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def to_dev(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=DRAM)


def dev_to_tor(t) -> torch.Tensor:
    return ttnn.to_torch(t).float()


# ---------------------------------------------------------------------------------------------
# Reference implementation (fp32 torch)
# ---------------------------------------------------------------------------------------------


def kda_full_reference(
    x, w_qkv, w_lowrank, taps, w_fb, w_gb, dt_bias, a_log, o_norm, w_o, lower_bound, eps, state=None, hist=None
):
    """Full KDA over ``x`` [T, hidden]: projections, conv, decay, recurrence, gated norm, o_proj."""
    T = x.shape[0]
    hist = torch.zeros(3, 3 * WIDTH) if hist is None else hist.clone()
    state = torch.zeros(H, K, K) if state is None else state.clone()

    # projections
    qkv_u = x @ w_qkv  # [T, 3HK]
    low = x @ w_lowrank  # [T, K + K + H]
    f_a, g_a, b = low.split([K, K, H], dim=-1)

    # conv + silu (tap j multiplies row t-3+j)
    window = torch.cat([hist, qkv_u], 0)
    conv = sum(window[t : t + T] * taps[j].unsqueeze(0) for j, t in enumerate(range(4)))
    qkv = F.silu(conv)
    new_hist = window[-3:]

    q, k, v = qkv.split(WIDTH, dim=-1)
    q, k, v = q.view(T, H, K), k.view(T, H, K), v.view(T, H, K)

    # decay
    f = f_a @ w_fb  # [T, HK]
    z = torch.exp(a_log.float()).repeat_interleave(K) * (f + dt_bias)  # per head AND channel
    g = lower_bound * torch.sigmoid(z)
    beta = torch.sigmoid(b)  # [T, H]

    # recurrence, fp32
    qn = F.normalize(q.float(), dim=-1, eps=1e-6) / math.sqrt(K)
    kn = F.normalize(k.float(), dim=-1, eps=1e-6)
    S = state.clone()
    o = torch.zeros(T, H, K)
    for t in range(T):
        S = torch.exp(g[t].view(H, K)).unsqueeze(-1) * S
        delta = (v[t].float() - torch.einsum("hk,hkv->hv", kn[t], S)) * beta[t].view(H, 1)
        S = S + kn[t].unsqueeze(-1) * delta.unsqueeze(1)
        o[t] = torch.einsum("hk,hkv->hv", qn[t], S)

    # gated RMSNorm over the head dim, then o_proj
    gate_out = g_a @ w_gb  # [T, HK]
    on = o / torch.sqrt(o.pow(2).mean(-1, keepdim=True) + eps) * o_norm.view(K)
    on = on * torch.sigmoid(gate_out.view(T, H, K))
    out = on.reshape(T, WIDTH) @ w_o
    return out, S, new_hist


# ---------------------------------------------------------------------------------------------
# Module construction
# ---------------------------------------------------------------------------------------------


def build_kda(device, seed=0, limit=10.0):
    torch.manual_seed(seed)
    w_qkv = torch.randn(HIDDEN, 3 * WIDTH) * 0.05
    w_lowrank = torch.randn(HIDDEN, 2 * K + H) * 0.05
    w_fb = torch.randn(K, WIDTH) * 0.05
    w_gb = torch.randn(K, WIDTH) * 0.05
    w_o = torch.randn(WIDTH, HIDDEN) * 0.05
    taps = torch.randn(4, 3 * WIDTH) * 0.2
    o_norm = torch.rand(K) + 0.5  # [head_dim] -- the module reshapes it to (1,1,1,head_dim)
    dt_bias = torch.randn(WIDTH) * 0.1
    a_log = torch.randn(H) * 0.1
    lower_bound = -5.0
    eps = 1e-5

    kda = KDA(
        device,
        w_qkv=to_dev(w_qkv, device, ttnn.bfloat8_b),
        w_lowrank=to_dev(w_lowrank, device, ttnn.bfloat16),
        w_fb=to_dev(w_fb, device, ttnn.bfloat16),
        w_gb=to_dev(w_gb, device, ttnn.bfloat16),
        w_o=to_dev(w_o, device, ttnn.bfloat8_b),
        taps=to_dev(taps, device, ttnn.bfloat16),
        o_norm=to_dev(o_norm, device, ttnn.bfloat16),
        dt_bias=dt_bias,
        a_log=a_log,
        num_heads=H,
        head_dim=K,
        hidden=HIDDEN,
        lower_bound=lower_bound,
        eps=eps,
    )
    ref = dict(
        w_qkv=w_qkv,
        w_lowrank=w_lowrank,
        taps=taps,
        w_fb=w_fb,
        w_gb=w_gb,
        dt_bias=dt_bias,
        a_log=a_log,
        o_norm=o_norm,
        w_o=w_o,
        lower_bound=lower_bound,
        eps=eps,
    )
    return kda, ref


# ---------------------------------------------------------------------------------------------
# 1. The port itself: with_carry == plain conv on one chip
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("T", [32, 64])
def test_conv_with_carry_matches_plain_conv(device, T):
    """The fused op the Blackhole->Wormhole port switched to must equal the plain conv op.

    On a single chip the sequence-parallel history exchange is a no-op, so
    ``qkv_causal_conv1d_silu_with_carry`` (4th output = replacement carry) has to reproduce
    ``qkv_causal_conv1d_silu`` exactly, and its carry has to be the same last-three-rows history the
    removed ``ChronologicalSelections.select_local_final_history`` used to produce.
    """
    from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import (
        qkv_device_inputs,
        qkv_reference,
    )

    widths = (128, 128, 128)  # 384 channels, a multiple of the 32-wide tile
    (t_in, t_hist, t_taps), (inp, hist, taps) = qkv_device_inputs(device, sequence=T, widths=widths)
    z = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int64),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    prog = ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=384)

    plain = ttnn.experimental.kda.qkv_causal_conv1d_silu(
        inp,
        hist,
        *taps,
        *widths,
        program_config=prog,
        actual_start=z,
        predecessor_carry=hist,
        memory_config=DRAM,
    )

    from models.demos.deepseek_v3_d_p.tt.kda.config import (
        KDA_HISTORY_EXCHANGE_ROWS,
        KDA_HISTORY_EXCHANGE_SEMAPHORES,
    )

    chans = sum(widths)
    exch_buf = ttnn.zeros(
        (1, KDA_HISTORY_EXCHANGE_ROWS, chans),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=DRAM,
    )
    core = ttnn._ttnn.operations.experimental.kda.qkv_causal_conv1d_silu_exchange_core(
        device.compute_with_storage_grid_size()
    )
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})
    sems = [
        ttnn.create_global_semaphore(device, cores, 0, ttnn.BufferType.L1_SMALL)
        for _ in range(KDA_HISTORY_EXCHANGE_SEMAPHORES)
    ]
    fused = ttnn.experimental.kda.qkv_causal_conv1d_silu_with_carry(
        inp,
        hist,
        *taps,
        *widths,
        program_config=prog,
        actual_start=z,
        actual_end=None,
        exchange_buffer=exch_buf,
        exchange_semaphores=sems,
        sequence_parallel_axis=0,
        memory_config=DRAM,
    )
    ttnn.synchronize_device(device)

    assert len(fused) == 4, f"with_carry must return q,k,v,carry, got {len(fused)}"
    for i in range(3):
        a = dev_to_tor(plain[i])
        b = dev_to_tor(fused[i])
        assert torch.equal(a, b), f"qkv[{i}] differs: max abs {float((a - b).abs().max())}"

    # And both agree with the torch reference.
    ref = qkv_reference(t_in, t_hist, t_taps, widths)
    for i in range(3):
        got = dev_to_tor(fused[i]).reshape(ref[i].shape)
        assert pcc(got, ref[i]) > 0.999, f"qkv[{i}] PCC {pcc(got, ref[i]):.6f}"

    # The carry is the three-row pre-conv history, in the incoming history's layout/shape.
    carry = fused[3]
    assert tuple(carry.shape) == tuple(hist.shape), f"carry {tuple(carry.shape)} != history {tuple(hist.shape)}"
    expected_hist = torch.cat([t_hist, t_in], 1)[:, -3:, :]
    assert pcc(dev_to_tor(carry).reshape(expected_hist.shape), expected_hist) > 0.999


# ---------------------------------------------------------------------------------------------
# 2. Whole-block prefill
# ---------------------------------------------------------------------------------------------
#
# BLACKHOLE -> WORMHOLE BLOCKER (unresolved; xfail rather than skipped so it stays visible)
#
# `prepare_chunk_recurrence` is the one KDA op that cannot run on Wormhole B0. Its Blackhole arch
# gate has been removed (device/prepare_chunk_recurrence_device_operation.cpp) and two latent bugs in
# this module's call were found and fixed on the way (beta must be BF16 and rank-3
# [1, num_chunks*32, num_heads], not fp32 [H, n_chunks, 32, 1]); the call now reaches kernel
# placement, where it fails on a hardware limit:
#
#   TT_THROW: ProgramSpec 'prepare_chunk_recurrence' places 33 DataflowBufferSpecs on node (3, 0),
#             but Gen1 supports at most 32 device slots per core
#
# The 33 buffers are the distinct per-core streams the chunk prep needs (q, k, v, g, beta, the
# beta/transpose pair, eye, tril, ones, gate_centered, gate_ones, block_masks, two workspaces, the
# centered inverse decay, akk, t_inv, v_beta, kd, q_decay, intra, final_decay, k_decay_transposed,
# anchor_decay, normalized_k, inverse_norms, ...). Every one of them is placed on every core that has
# work, so giving the op more cores does not help: the count per core is what exceeds the limit.
#
# Lifting it is kernel surgery, not a port: the 33 streams have to be aliased or sequenced so at most
# 32 are live at once per core. Two directions:
#   * reuse buffers whose lifetimes do not overlap (e.g. `eye`/`tril`/`ones`/`block_masks` are
#     constants consumed early, and `workspace_0`/`workspace_1` are disjoint in time);
#   * split the op: run the `T_inv` construction as its own program and keep only its output live,
#     which removes the workspaces and masks from the prep program entirely.
# Until then, KDA *prefill* is Blackhole-only; KDA *decode* (a different op path) works on Wormhole,
# see test_kda_decode_matches_reference below.
PREP_BLOCKER = (
    "prepare_chunk_recurrence needs 33 DataflowBufferSpecs per core but Wormhole Gen1 allows 32; "
    "lifting it requires aliasing/sequencing the kernel's buffers (see the comment above)"
)


@pytest.mark.xfail(strict=True, reason=PREP_BLOCKER)
@pytest.mark.parametrize("T", [32])
def test_kda_prefill_matches_reference(device, T):
    """One prefill chunk of the whole block, against the fp32 torch reference."""
    kda, ref = build_kda(device)
    state = kda.allocate_state(1)
    x = torch.randn(T, HIDDEN) * 0.5

    xt = to_dev(x.reshape(1, 1, T, HIDDEN), device, ttnn.bfloat16)
    actual_end = ttnn.from_torch(
        torch.tensor([T], dtype=torch.int64), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    valid_end_f = ttnn.from_torch(
        torch.tensor([float(T)], dtype=torch.float32),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    out = kda.prefill(xt, state, 0, actual_end=actual_end, valid_end_f=valid_end_f)
    ttnn.synchronize_device(device)

    ref_out, ref_state, _ = kda_full_reference(x, **ref)
    got = dev_to_tor(out).reshape(T, HIDDEN)
    assert pcc(got, ref_out) > 0.98, f"prefill PCC {pcc(got, ref_out):.5f}"

    got_state = dev_to_tor(state.recurrent[0]).reshape(H, K, K)
    assert pcc(got_state, ref_state) > 0.98, f"recurrent state PCC {pcc(got_state, ref_state):.5f}"


@pytest.mark.xfail(strict=True, reason=PREP_BLOCKER)
def test_kda_prefill_carries_history_across_chunks(device):
    """Two chunks must equal one long chunk: this is what the conv-history carry protects, and it is
    exactly the path the port changed."""
    kda, ref = build_kda(device, seed=1)
    T = 32
    x = torch.randn(2 * T, HIDDEN) * 0.5

    # Two chunks, carrying state and history.
    state = kda.allocate_state(1)
    outs = []
    for c in range(2):
        xt = to_dev(x[c * T : (c + 1) * T].reshape(1, 1, T, HIDDEN), device, ttnn.bfloat16)
        ae = ttnn.from_torch(
            torch.tensor([T], dtype=torch.int64), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        ve = ttnn.from_torch(
            torch.tensor([float(T)], dtype=torch.float32),
            device=device,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        o = kda.prefill(xt, state, 0, actual_end=ae, valid_end_f=ve)
        ttnn.synchronize_device(device)
        outs.append(dev_to_tor(o).reshape(T, HIDDEN))

    # One unchunked pass.
    ref_out, _, _ = kda_full_reference(x, **ref)
    got = torch.cat(outs, 0)
    assert pcc(got, ref_out) > 0.98, f"chunked-vs-single PCC {pcc(got, ref_out):.5f}"
    # And the first chunk must already match, so a carry bug cannot hide in the average.
    assert pcc(outs[0], ref_out[:T]) > 0.98


# ---------------------------------------------------------------------------------------------
# 3. Decode
# ---------------------------------------------------------------------------------------------


def test_kda_decode_matches_reference(device):
    """Single-token decode steps, state carried in the persistent slot."""
    kda, ref = build_kda(device, seed=2)
    state = kda.allocate_state(1)
    T = 4
    x = torch.randn(T, HIDDEN) * 0.5

    outs = []
    for t in range(T):
        xt = to_dev(x[t : t + 1].reshape(1, 1, 1, HIDDEN), device, ttnn.bfloat16)
        o = kda.decode(xt, state)
        ttnn.synchronize_device(device)
        outs.append(dev_to_tor(o).reshape(HIDDEN))

    ref_out, _, _ = kda_full_reference(x, **ref)
    got = torch.stack(outs, 0)
    assert pcc(got, ref_out) > 0.98, f"decode PCC {pcc(got, ref_out):.5f}"


@pytest.mark.xfail(strict=True, reason=PREP_BLOCKER)
def test_prefill_then_decode_agrees(device):
    """A prefill chunk followed by decode steps must equal a single reference pass: the strongest
    single check that the carried state and conv history are consistent between the two paths."""
    kda, ref = build_kda(device, seed=3)
    state = kda.allocate_state(1)
    T = 32
    D = 4
    x = torch.randn(T + D, HIDDEN) * 0.5

    xt = to_dev(x[:T].reshape(1, 1, T, HIDDEN), device, ttnn.bfloat16)
    ae = ttnn.from_torch(
        torch.tensor([T], dtype=torch.int64), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    ve = ttnn.from_torch(
        torch.tensor([float(T)], dtype=torch.float32),
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    pre = kda.prefill(xt, state, 0, actual_end=ae, valid_end_f=ve)
    ttnn.synchronize_device(device)

    dec = []
    for t in range(T, T + D):
        o = kda.decode(to_dev(x[t : t + 1].reshape(1, 1, 1, HIDDEN), device, ttnn.bfloat16), state)
        ttnn.synchronize_device(device)
        dec.append(dev_to_tor(o).reshape(HIDDEN))

    ref_out, _, _ = kda_full_reference(x, **ref)
    got_pre = dev_to_tor(pre).reshape(T, HIDDEN)
    got = torch.cat([got_pre, torch.stack(dec, 0)], 0)
    assert pcc(got, ref_out) > 0.97, f"prefill+decode PCC {pcc(got, ref_out):.5f}"
    # The decode positions are where a carry bug shows up.
    assert pcc(torch.stack(dec, 0), ref_out[T:]) > 0.97, f"decode-tail PCC {pcc(torch.stack(dec, 0), ref_out[T:]):.5f}"


# ---------------------------------------------------------------------------------------------
# 4. State / bookkeeping
# ---------------------------------------------------------------------------------------------


def test_allocate_state_shapes(device):
    """Per-slot state: recurrent [1,H,K,K] fp32 and conv history [1,3,3HK] bf16 ROW_MAJOR."""
    kda, _ = build_kda(device)
    st = kda.allocate_state(2)
    assert isinstance(st, KDAStateCache)
    assert st.slots == 2
    assert tuple(st.recurrent[0].shape) == (1, H, K, K)
    assert st.recurrent[0].dtype == KDA_STATE_DTYPE
    assert tuple(st.conv[0].shape) == (1, CONV_K - 1, 3 * WIDTH)
    assert st.conv[0].layout == ttnn.ROW_MAJOR_LAYOUT


def test_fresh_state_is_zero(device):
    """A freshly allocated state must be zeros: decode from position 0 depends on it."""
    kda, _ = build_kda(device)
    st = kda.allocate_state(1)
    assert float(dev_to_tor(st.recurrent[0]).abs().max()) == 0.0
    assert float(dev_to_tor(st.conv[0]).abs().max()) == 0.0


def test_state_is_updated_in_place(device):
    """The persistent buffers must keep their addresses (a captured trace depends on it)."""
    kda, _ = build_kda(device, seed=4)
    st = kda.allocate_state(1)
    addr_rec = st.recurrent[0].buffer_address()
    addr_conv = st.conv[0].buffer_address()
    x = torch.randn(1, HIDDEN) * 0.5
    kda.decode(to_dev(x.reshape(1, 1, 1, HIDDEN), device, ttnn.bfloat16), st)
    ttnn.synchronize_device(device)
    assert st.recurrent[0].buffer_address() == addr_rec
    assert st.conv[0].buffer_address() == addr_conv
    assert float(dev_to_tor(st.recurrent[0]).abs().max()) > 0.0, "state should have changed"
