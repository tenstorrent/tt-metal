# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC and perf for fused recurrent gated-delta-rule kernels against the FLA naive reference.
T=1 is decode; T=K+1 returns per-token state for spec-decode verify."""
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.gdn_fla_ref import (
    l2norm_fla,
    make_gdn_inputs,
    naive_recurrent_gated_delta_rule,
    naive_recurrent_per_token_state,
    pcc,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    fused_recurrent_gated_delta_rule_ttnn,
    recurrent_gated_delta_rule_decode_ttnn,
)

# Qwen3.6-27B GDN: H=num_value_heads=32, Dk=Dv=128. The op is per-head, so full H is the same math.
H, DK, DV = 32, 128, 128
SCALE = DK**-0.5

FUSED_OP = getattr(getattr(ttnn, "transformer", None), "fused_recurrent_gated_delta_rule", None)
_needs_op = pytest.mark.skipif(FUSED_OP is None, reason="fused_recurrent_gated_delta_rule op not built yet")


def _to_dev(mesh_device, t, dtype=ttnn.float32):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _dev0(mesh_device, t, first=1):
    """Read the device-0 copy of a replicated output: concat over dim0, take the first `first` rows."""
    full = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
    return full[:first]


def _ref_decode(q, k, v, beta, g, s0):
    """FLA-naive T=1 step with the op's internal L2-norm applied to q/k (contract in gdn_fla_ref)."""
    o, h = naive_recurrent_gated_delta_rule(
        l2norm_fla(q), l2norm_fla(k), v, beta, g, scale=SCALE, initial_state=s0, output_final_state=True
    )
    return o, h


def _time_op(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(_time_op.mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(_time_op.mesh)
    return (time.perf_counter() - t0) / iters * 1e3


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_composite_decode_matches_fla_naive(mesh_device, seed):
    """Composite decode vs FLA naive: in-kernel L2-norm, q-scale, and exp(g) decay."""
    q, k, v, beta, g = make_gdn_inputs(T=1, H=H, Dk=DK, Dv=DV, seed=seed)
    s0 = torch.randn(1, H, DK, DV, dtype=torch.float32) * 0.1

    o_ref, h_ref = _ref_decode(q, k, v, beta, g, s0)

    q_tt, k_tt, v_tt = (_to_dev(mesh_device, x) for x in (q, k, v))
    beta_tt = _to_dev(mesh_device, beta)
    g_tt = _to_dev(mesh_device, g)
    s0_tt = _to_dev(mesh_device, s0)

    o_tt, h_tt = recurrent_gated_delta_rule_decode_ttnn(
        q_tt, k_tt, v_tt, beta_tt, g_tt, scale=SCALE, initial_state=s0_tt, device=mesh_device, high_precision=True
    )
    o_dev = _dev0(mesh_device, o_tt)
    h_dev = _dev0(mesh_device, h_tt)

    p_o = pcc(o_dev, o_ref)
    p_h = pcc(h_dev, h_ref)
    logger.info(f"[composite decode] seed={seed} PCC o={p_o:.6f} state={p_h:.6f}")
    assert p_o > 0.999, f"output PCC {p_o} below 0.999"
    assert p_h > 0.999, f"state PCC {p_h} below 0.999"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_composite_decode_baseline_perf(mesh_device):
    """Baseline per-step latency of the composite decode op (the number the fused op must beat)."""
    _time_op.mesh = mesh_device
    q, k, v, beta, g = make_gdn_inputs(T=1, H=H, Dk=DK, Dv=DV, seed=0)
    s0 = torch.zeros(1, H, DK, DV, dtype=torch.float32)
    q_tt, k_tt, v_tt = (_to_dev(mesh_device, x) for x in (q, k, v))
    beta_tt, g_tt, s0_tt = _to_dev(mesh_device, beta), _to_dev(mesh_device, g), _to_dev(mesh_device, s0)

    def step():
        o, h = recurrent_gated_delta_rule_decode_ttnn(
            q_tt, k_tt, v_tt, beta_tt, g_tt, scale=SCALE, initial_state=s0_tt, device=mesh_device, high_precision=True
        )
        ttnn.deallocate(o)
        ttnn.deallocate(h)

    ms = _time_op(step)
    logger.info(f"[composite decode] baseline latency = {ms:.4f} ms/step (H={H}, Dk={DK}, Dv={DV})")


@_needs_op
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fused_decode_matches_fla_naive(mesh_device, seed):
    """Kernel 1: fused device op, T=1, vs FLA naive."""
    q, k, v, beta, g = make_gdn_inputs(T=1, H=H, Dk=DK, Dv=DV, seed=seed)
    s0 = torch.randn(1, H, DK, DV, dtype=torch.float32) * 0.1
    o_ref, h_ref = _ref_decode(q, k, v, beta, g, s0)

    q_tt, k_tt, v_tt = (_to_dev(mesh_device, x) for x in (q, k, v))
    beta_tt, g_tt, s0_tt = _to_dev(mesh_device, beta), _to_dev(mesh_device, g), _to_dev(mesh_device, s0)
    o_tt, h_tt = fused_recurrent_gated_delta_rule_ttnn(
        q_tt, k_tt, v_tt, beta_tt, g_tt, scale=SCALE, initial_state=s0_tt, device=mesh_device
    )
    p_o = pcc(_dev0(mesh_device, o_tt), o_ref)
    p_h = pcc(_dev0(mesh_device, h_tt), h_ref)
    logger.info(f"[fused decode] seed={seed} PCC o={p_o:.6f} state={p_h:.6f}")
    assert p_o > 0.999 and p_h > 0.999


@_needs_op
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fused_decode_no_initial_state(mesh_device, seed):
    """T=1 with no initial_state must materialize a zeros buffer; the reader reads S unconditionally."""
    q, k, v, beta, g = make_gdn_inputs(T=1, H=H, Dk=DK, Dv=DV, seed=seed)
    s0_zero = torch.zeros(1, H, DK, DV, dtype=torch.float32)
    o_ref, h_ref = _ref_decode(q, k, v, beta, g, s0_zero)

    q_tt, k_tt, v_tt = (_to_dev(mesh_device, x) for x in (q, k, v))
    beta_tt, g_tt = _to_dev(mesh_device, beta), _to_dev(mesh_device, g)
    o_tt, h_tt = fused_recurrent_gated_delta_rule_ttnn(
        q_tt, k_tt, v_tt, beta_tt, g_tt, scale=SCALE, initial_state=None, device=mesh_device
    )
    p_o = pcc(_dev0(mesh_device, o_tt), o_ref)
    p_h = pcc(_dev0(mesh_device, h_tt), h_ref)
    logger.info(f"[fused decode, no s0] seed={seed} PCC o={p_o:.6f} state={p_h:.6f}")
    assert p_o > 0.999, f"output PCC {p_o} below 0.999"
    assert p_h > 0.999, f"state PCC {p_h} below 0.999"


@_needs_op
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_fused_decode_perf(mesh_device):
    """Kernel 1 latency: fused op vs the composite baseline, same inputs/dtype. Reports both."""
    _time_op.mesh = mesh_device
    q, k, v, beta, g = make_gdn_inputs(T=1, H=H, Dk=DK, Dv=DV, seed=0)
    s0 = torch.zeros(1, H, DK, DV, dtype=torch.float32)
    q_tt, k_tt, v_tt = (_to_dev(mesh_device, x) for x in (q, k, v))
    beta_tt, g_tt, s0_tt = _to_dev(mesh_device, beta), _to_dev(mesh_device, g), _to_dev(mesh_device, s0)

    def composite():
        o, h = recurrent_gated_delta_rule_decode_ttnn(
            q_tt, k_tt, v_tt, beta_tt, g_tt, scale=SCALE, initial_state=s0_tt, device=mesh_device, high_precision=True
        )
        ttnn.deallocate(o)
        ttnn.deallocate(h)

    def fused():
        o, h = fused_recurrent_gated_delta_rule_ttnn(
            q_tt, k_tt, v_tt, beta_tt, g_tt, scale=SCALE, initial_state=s0_tt, device=mesh_device
        )
        ttnn.deallocate(o)
        ttnn.deallocate(h)

    ms_c = _time_op(composite)
    ms_f = _time_op(fused)
    logger.info(f"[decode perf] composite={ms_c:.4f} ms  fused={ms_f:.4f} ms  speedup={ms_c/ms_f:.2f}x")


def _seq_composite_decode(mesh_device, q, k, v, beta, g, s0):
    """Token-by-token composite decode, threading state. Returns per-token output and state."""
    T = q.shape[1]
    s_tt = _to_dev(mesh_device, s0)
    outs, states = [], []
    for t in range(T):
        qt = _to_dev(mesh_device, q[:, t : t + 1])
        kt = _to_dev(mesh_device, k[:, t : t + 1])
        vt = _to_dev(mesh_device, v[:, t : t + 1])
        bt = _to_dev(mesh_device, beta[:, t : t + 1])
        gt = _to_dev(mesh_device, g[:, t : t + 1])
        o_t, s_tt = recurrent_gated_delta_rule_decode_ttnn(
            qt, kt, vt, bt, gt, scale=SCALE, initial_state=s_tt, device=mesh_device, high_precision=True
        )
        outs.append(_dev0(mesh_device, o_t))
        states.append(_dev0(mesh_device, s_tt))
    return outs, states


@_needs_op
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("K", [3, 4])
@pytest.mark.parametrize("seed", [0, 1])
def test_fused_verify_matches_fla_naive(mesh_device, K, seed):
    """Fused T=K+1 op vs FLA naive, including the state after every token."""
    T = K + 1
    q, k, v, beta, g = make_gdn_inputs(T=T, H=H, Dk=DK, Dv=DV, seed=seed)
    s0 = torch.randn(1, H, DK, DV, dtype=torch.float32) * 0.1

    o_ref, st_ref = naive_recurrent_per_token_state(
        l2norm_fla(q), l2norm_fla(k), v, beta, g, scale=SCALE, initial_state=s0
    )

    q_tt, k_tt, v_tt = (_to_dev(mesh_device, x) for x in (q, k, v))
    beta_tt, g_tt, s0_tt = _to_dev(mesh_device, beta), _to_dev(mesh_device, g), _to_dev(mesh_device, s0)
    o_tt, st_tt = fused_recurrent_gated_delta_rule_ttnn(
        q_tt,
        k_tt,
        v_tt,
        beta_tt,
        g_tt,
        scale=SCALE,
        initial_state=s0_tt,
        device=mesh_device,
        output_per_token_state=True,
    )
    o_dev = _dev0(mesh_device, o_tt)
    st_dev = _dev0(mesh_device, st_tt)

    p_o = pcc(o_dev, o_ref)
    p_s = pcc(st_dev, st_ref)
    per_tok = [pcc(st_dev[:, t], st_ref[:, t]) for t in range(T)]
    logger.info(
        f"[fused verify] K={K} seed={seed} PCC o={p_o:.6f} state={p_s:.6f} per-tok-state={[f'{x:.5f}' for x in per_tok]}"
    )
    assert p_o > 0.999, f"output PCC {p_o}"
    assert p_s > 0.999, f"state PCC {p_s}"
    assert min(per_tok) > 0.999, f"per-token state min PCC {min(per_tok)}"


@_needs_op
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("K", [3, 4])
def test_fused_verify_matches_sequential_decode(mesh_device, K):
    """Fused per-token output and state must match the sequential composite recurrence."""
    T = K + 1
    q, k, v, beta, g = make_gdn_inputs(T=T, H=H, Dk=DK, Dv=DV, seed=7)
    s0 = torch.randn(1, H, DK, DV, dtype=torch.float32) * 0.1

    seq_o, seq_s = _seq_composite_decode(mesh_device, q, k, v, beta, g, s0)

    q_tt, k_tt, v_tt = (_to_dev(mesh_device, x) for x in (q, k, v))
    beta_tt, g_tt, s0_tt = _to_dev(mesh_device, beta), _to_dev(mesh_device, g), _to_dev(mesh_device, s0)
    o_tt, st_tt = fused_recurrent_gated_delta_rule_ttnn(
        q_tt,
        k_tt,
        v_tt,
        beta_tt,
        g_tt,
        scale=SCALE,
        initial_state=s0_tt,
        device=mesh_device,
        output_per_token_state=True,
    )
    o_dev = _dev0(mesh_device, o_tt)
    st_dev = _dev0(mesh_device, st_tt)

    for t in range(T):
        p_o = pcc(o_dev[:, t].reshape(1, 1, H, DV), seq_o[t])
        p_s = pcc(st_dev[:, t], seq_s[t])
        logger.info(f"[verify vs seq] K={K} t={t} PCC o={p_o:.6f} state={p_s:.6f}")
        assert p_o > 0.999 and p_s > 0.999


@_needs_op
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("K", [3, 4])
def test_fused_verify_perf(mesh_device, K):
    """Fused one-dispatch latency vs T sequential composite decode calls."""
    _time_op.mesh = mesh_device
    T = K + 1
    q, k, v, beta, g = make_gdn_inputs(T=T, H=H, Dk=DK, Dv=DV, seed=0)
    s0 = torch.zeros(1, H, DK, DV, dtype=torch.float32)

    qk_seq = [
        (
            _to_dev(mesh_device, q[:, t : t + 1]),
            _to_dev(mesh_device, k[:, t : t + 1]),
            _to_dev(mesh_device, v[:, t : t + 1]),
            _to_dev(mesh_device, beta[:, t : t + 1]),
            _to_dev(mesh_device, g[:, t : t + 1]),
        )
        for t in range(T)
    ]
    s0_tt = _to_dev(mesh_device, s0)
    q_tt, k_tt, v_tt = (_to_dev(mesh_device, x) for x in (q, k, v))
    beta_tt, g_tt = _to_dev(mesh_device, beta), _to_dev(mesh_device, g)

    def sequential():
        s = s0_tt
        for qt, kt, vt, bt, gt in qk_seq:
            o, s = recurrent_gated_delta_rule_decode_ttnn(
                qt, kt, vt, bt, gt, scale=SCALE, initial_state=s, device=mesh_device, high_precision=True
            )
            ttnn.deallocate(o)

    def fused():
        o, st = fused_recurrent_gated_delta_rule_ttnn(
            q_tt,
            k_tt,
            v_tt,
            beta_tt,
            g_tt,
            scale=SCALE,
            initial_state=s0_tt,
            device=mesh_device,
            output_per_token_state=True,
        )
        ttnn.deallocate(o)
        ttnn.deallocate(st)

    ms_seq = _time_op(sequential)
    ms_f = _time_op(fused)
    logger.info(
        f"[verify perf] K={K} (T={T})  sequential={ms_seq:.4f} ms  fused={ms_f:.4f} ms  speedup={ms_seq/ms_f:.2f}x"
    )
