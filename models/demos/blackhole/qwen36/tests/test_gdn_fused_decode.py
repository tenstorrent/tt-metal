# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused GDN decode ops vs the composite path and a torch fp32 golden.

Covers both TP head configs of Qwen3.6/3.8-27B GDN (global 16 K / 48 V heads @128):
TP=8 -> per-device Nk=2/Nv=6, TP=4 -> Nk=4/Nv=12. B=8 throughout.

Single-device tests: the fused ops are pure per-device programs (no CCL), so one
card exercises exactly what each mesh device runs.
"""

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.fused_decode import op as fused_op
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_decode_ttnn,
)
from models.common.utility_functions import comp_pcc

B = 8
DK = DV = 128
K_TAPS = 4
EPS = 1e-6


def _dims(nk, nv):
    kd = nk * DK
    qkv_dim = 2 * kd + nv * DV
    z_dim = nv * DV
    qkvzab_dim = qkv_dim + z_dim + 2 * nv
    return kd, qkv_dim, z_dim, qkvzab_dim


def _dev(device, t, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def _rand_inputs(nk, nv, seed):
    torch.manual_seed(seed)
    kd, qkv_dim, z_dim, qkvzab_dim = _dims(nk, nv)
    qkvzab = torch.randn(1, B, qkvzab_dim, dtype=torch.float32) * 0.5
    states = [torch.randn(1, B, qkv_dim, dtype=torch.float32) * 0.5 for _ in range(K_TAPS)]
    taps = [torch.randn(1, 1, qkv_dim, dtype=torch.float32) * 0.5 for _ in range(K_TAPS)]
    h0 = torch.randn(B, nv, DK, DV, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(1, 1, nv, dtype=torch.float32) * 0.5
    neg_exp_a = -torch.exp(torch.randn(1, 1, nv, dtype=torch.float32) * 0.3)
    norm_w = torch.randn(1, 1, DV, dtype=torch.float32) * 0.5 + 1.0
    return qkvzab, states, taps, h0, dt_bias, neg_exp_a, norm_w


def _bf16(t):
    return t.to(torch.bfloat16).float()


def golden_conv(qkvzab, states, taps, qkv_dim):
    """Composite conv semantics: shift register then silu(sum tap_j * st_j), in bf16 inputs."""
    x_in = _bf16(qkvzab)[..., :qkv_dim]
    window = [_bf16(states[1]), _bf16(states[2]), _bf16(states[3]), x_in]
    conv = sum(_bf16(taps[j]) * window[j] for j in range(K_TAPS))
    conv = torch.nn.functional.silu(conv)
    return conv, window  # window == the shifted states


def golden_recurrence(conv, qkvzab, h0, dt_bias, neg_exp_a, norm_w, nk, nv):
    kd, qkv_dim, z_dim, _ = _dims(nk, nv)
    rf = nv // nk
    conv = conv.reshape(B, qkv_dim)
    q = conv[:, :kd].reshape(B, nk, DK).repeat_interleave(rf, dim=1)
    k = conv[:, kd : 2 * kd].reshape(B, nk, DK).repeat_interleave(rf, dim=1)
    v = conv[:, 2 * kd :].reshape(B, nv, DV)
    zab = _bf16(qkvzab).reshape(B, -1)
    z = zab[:, qkv_dim : qkv_dim + z_dim].reshape(B, nv, DV)
    a = zab[:, qkv_dim + z_dim : qkv_dim + z_dim + nv]
    b_ = zab[:, qkv_dim + z_dim + nv : qkv_dim + z_dim + 2 * nv]

    beta = torch.sigmoid(b_)
    g = _bf16(neg_exp_a).reshape(nv) * torch.nn.functional.softplus(a + _bf16(dt_bias).reshape(nv), 1.0, 20.0)
    decay = torch.exp(g)

    qn = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + EPS) * DK**-0.5
    kn = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + EPS)

    hd = h0 * decay[..., None, None]
    v_read = torch.einsum("bhk,bhkv->bhv", kn, hd)
    delta = (v - v_read) * beta[..., None]
    h_new = hd + torch.einsum("bhk,bhv->bhkv", kn, delta)
    o = torch.einsum("bhk,bhkv->bhv", qn, h_new)

    out = o / torch.sqrt(o.pow(2).mean(-1, keepdim=True) + EPS)
    out = out * _bf16(norm_w).reshape(DV) * torch.nn.functional.silu(z)
    return out.reshape(1, B, nv * DV), h_new


def composite_recurrence_ttnn(device, conv, qkvzab, h0, dt_bias, neg_exp_a, norm_w, nk, nv):
    """The tp.py forward_decode chain from the conv output to the gated output."""
    kd, qkv_dim, z_dim, _ = _dims(nk, nv)
    rf = nv // nk
    conv_d = _dev(device, conv.reshape(1, B, qkv_dim))
    q = ttnn.reshape(ttnn.slice(conv_d, (0, 0, 0), (1, B, kd)), (B, nk, DK))
    k = ttnn.reshape(ttnn.slice(conv_d, (0, 0, kd), (1, B, 2 * kd)), (B, nk, DK))
    v = ttnn.reshape(ttnn.slice(conv_d, (0, 0, 2 * kd), (1, B, qkv_dim)), (B, nv, DV))
    q = ttnn.repeat_interleave(q, rf, dim=1)
    k = ttnn.repeat_interleave(k, rf, dim=1)
    q = ttnn.reshape(q, (B, 1, nv, DK))
    k = ttnn.reshape(k, (B, 1, nv, DK))
    v = ttnn.reshape(v, (B, 1, nv, DV))

    zab = _bf16(qkvzab).reshape(B, -1)
    z = _dev(device, zab[:, qkv_dim : qkv_dim + z_dim].reshape(1, B, z_dim))
    a = _dev(device, zab[:, qkv_dim + z_dim : qkv_dim + z_dim + nv].reshape(1, B, nv))
    b_ = _dev(device, zab[:, qkv_dim + z_dim + nv : qkv_dim + z_dim + 2 * nv].reshape(1, B, nv))
    dtb = _dev(device, dt_bias)
    nega = _dev(device, neg_exp_a)
    w = _dev(device, norm_w)

    beta = ttnn.reshape(ttnn.sigmoid(b_), (B, 1, nv))
    g = ttnn.multiply(nega, ttnn.add(a, dtb, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)]))
    g = ttnn.reshape(g, (B, 1, nv))

    h0_d = _dev(device, h0, dtype=ttnn.float32)
    o, new_h = recurrent_gated_delta_rule_decode_ttnn(
        q, k, v, beta, g, scale=DK**-0.5, initial_state=h0_d, device=device, high_precision=True
    )

    out_r = ttnn.reshape(o, (B, nv, DV))
    out_n = ttnn.rms_norm(out_r, weight=w, epsilon=1e-6)
    out_f = ttnn.reshape(out_n, (1, B, nv * DV))
    gated = ttnn.multiply(out_f, ttnn.silu(z))
    return ttnn.to_torch(gated).float(), ttnn.to_torch(new_h).float()


@pytest.mark.parametrize("nk,nv", [(2, 6), (4, 12)], ids=["tp8", "tp4"])
@pytest.mark.parametrize("qkvzab_mc", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
def test_conv_shift_silu(device, nk, nv, qkvzab_mc):
    kd, qkv_dim, _, _ = _dims(nk, nv)
    qkvzab, states, taps, _, _, _, _ = _rand_inputs(nk, nv, seed=0)

    qkvzab_d = _dev(device, qkvzab, memory_config=qkvzab_mc)
    states_d = [_dev(device, s) for s in states]
    taps_d = [_dev(device, t) for t in taps]
    conv_out_d = _dev(device, torch.zeros(1, B, qkv_dim))

    fused_op.conv_shift_silu(qkvzab_d, states_d, taps_d, conv_out_d)

    conv_ref, shifted_ref = golden_conv(qkvzab, states, taps, qkv_dim)
    ok, pcc = comp_pcc(conv_ref, ttnn.to_torch(conv_out_d).float(), 0.999)
    assert ok, f"conv PCC {pcc}"
    for j in range(K_TAPS):
        got = ttnn.to_torch(states_d[j]).float()
        assert torch.equal(got, shifted_ref[j]), f"shift register writeback mismatch at tap {j}"


@pytest.mark.parametrize("nk,nv", [(2, 6), (4, 12)], ids=["tp8", "tp4"])
def test_recurrence(device, nk, nv):
    kd, qkv_dim, _, _ = _dims(nk, nv)
    qkvzab, states, taps, h0, dt_bias, neg_exp_a, norm_w = _rand_inputs(nk, nv, seed=1)
    conv_ref, _ = golden_conv(qkvzab, states, taps, qkv_dim)

    conv_d = _dev(device, conv_ref.reshape(1, B, qkv_dim))
    qkvzab_d = _dev(device, qkvzab)
    state_d = _dev(device, h0, dtype=ttnn.float32)
    dtb_d = _dev(device, dt_bias)
    nega_d = _dev(device, neg_exp_a)
    w_d = _dev(device, norm_w)
    out_d = _dev(device, torch.zeros(1, B, nv * DV), dtype=ttnn.float32)

    fused_op.recurrence(conv_d, qkvzab_d, state_d, dtb_d, nega_d, w_d, out_d, nk=nk, nv=nv, dk=DK, dv=DV, b_rows=B)

    out_ref, h_ref = golden_recurrence(_bf16(conv_ref), qkvzab, h0, dt_bias, neg_exp_a, norm_w, nk, nv)
    out_got = ttnn.to_torch(out_d).float()
    h_got = ttnn.to_torch(state_d).float()
    ok, pcc = comp_pcc(out_ref, out_got, 0.999)
    assert ok, f"gated output PCC vs torch golden: {pcc}"
    ok, pcc = comp_pcc(h_ref, h_got, 0.999)
    assert ok, f"state PCC vs torch golden: {pcc}"

    out_comp, h_comp = composite_recurrence_ttnn(device, conv_ref, qkvzab, h0, dt_bias, neg_exp_a, norm_w, nk, nv)
    ok, pcc = comp_pcc(out_comp, out_got, 0.999)
    assert ok, f"gated output PCC vs composite ttnn: {pcc}"
    ok, pcc = comp_pcc(h_comp.reshape(h_got.shape), h_got, 0.999)
    assert ok, f"state PCC vs composite ttnn: {pcc}"


@pytest.mark.parametrize("nk,nv", [(2, 6), (4, 12)], ids=["tp8", "tp4"])
def test_state_evolution(device, nk, nv):
    """N sequential fused steps track the torch golden trajectory (fp32 state in place)."""
    n_steps = 8
    kd, qkv_dim, _, _ = _dims(nk, nv)
    _, states, taps, h0, dt_bias, neg_exp_a, norm_w = _rand_inputs(nk, nv, seed=2)

    states_ref = [s.clone() for s in states]
    h_ref = h0.clone()

    states_d = [_dev(device, s) for s in states]
    taps_d = [_dev(device, t) for t in taps]
    state_d = _dev(device, h0, dtype=ttnn.float32)
    dtb_d = _dev(device, dt_bias)
    nega_d = _dev(device, neg_exp_a)
    w_d = _dev(device, norm_w)
    conv_out_d = _dev(device, torch.zeros(1, B, qkv_dim))
    out_d = _dev(device, torch.zeros(1, B, nv * DV), dtype=ttnn.float32)

    for step in range(n_steps):
        torch.manual_seed(100 + step)
        qkvzab = torch.randn(1, B, _dims(nk, nv)[3], dtype=torch.float32) * 0.5
        qkvzab_d = _dev(device, qkvzab)

        fused_op.conv_shift_silu(qkvzab_d, states_d, taps_d, conv_out_d)
        fused_op.recurrence(
            conv_out_d, qkvzab_d, state_d, dtb_d, nega_d, w_d, out_d, nk=nk, nv=nv, dk=DK, dv=DV, b_rows=B
        )
        ttnn.deallocate(qkvzab_d)

        conv_ref, shifted = golden_conv(qkvzab, states_ref, taps, qkv_dim)
        states_ref = [s.clone() for s in shifted]
        out_ref, h_ref = golden_recurrence(_bf16(conv_ref), qkvzab, h_ref, dt_bias, neg_exp_a, norm_w, nk, nv)

        ok, pcc = comp_pcc(h_ref, ttnn.to_torch(state_d).float(), 0.999)
        assert ok, f"step {step}: state trajectory PCC {pcc}"
        ok, pcc = comp_pcc(out_ref, ttnn.to_torch(out_d).float(), 0.999)
        assert ok, f"step {step}: gated output PCC {pcc}"
