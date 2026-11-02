# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_1C_GDN_GB: the recurrence op computing decay = exp(-exp(A_log) * softplus(a + dt_bias)) and
beta = sigmoid(b) from the raw a|b row (fused_gating), against today's host chain, both scored against a float64
reference built from the same bf16 inputs. The fused path is fp32 throughout, so it is not bit-exact with the bf16
chain. The asserts are on two probes that isolate the gating: with k = 0 the step is S*decay (each head's
effective decay), with S = 0 it is beta * (k (x) v) (each head's effective beta); the fused values must be at least
as close to float64 as the chain's. The full step is reported, not asserted on o: the op rounds q internally at
about bf16, so a max-error comparison of o mostly measures that (A's 23:05 run: o 5.6e-3 vs 4.6e-3 while the
fused state was closer, 5.9e-3 vs 6.8e-3).

    pytest models/demos/blackhole/bonsai2/tp1/test_gdn_gb.py -s
"""
import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.tp import _softplus_add, fused_recurrent_trim

NK, NV, D = 16, 48, 128


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_gdn_gb(mesh_device):
    torch.manual_seed(0)
    L1 = ttnn.L1_MEMORY_CONFIG
    f32 = ttnn.float32

    def dev(t, dt=ttnn.bfloat16):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=L1)

    bf = torch.bfloat16
    q_t, k_t, v_t = (torch.randn(1, 1, n, D).to(bf) for n in (NK, NK, NV))
    ab_t = (torch.randn(1, 1, 2 * NV) * 2).to(bf)
    a_log_t = (torch.rand(NV) * 2 - 1).to(bf)  # as loaded: A_log and dt_bias in bf16
    dt_t = (torch.randn(NV) * 0.5).to(bf)
    s_t = torch.randn(NV, D, D) * 0.1

    neg_exp_a = ttnn.neg(ttnn.exp(dev(a_log_t.float().reshape(1, 1, NV))))  # as tp.py builds tw["neg_exp_A"]
    dt = dev(dt_t.float().reshape(1, 1, NV))
    q, k, v = dev(q_t), dev(k_t), dev(v_t)

    # today's chain (TRIM): slices of the a|b row, sigmoid, softplus-add, multiply to fp32
    ab = dev(ab_t)
    a, b = ttnn.slice(ab, (0, 0, 0), (1, 1, NV)), ttnn.slice(ab, (0, 0, NV), (1, 1, 2 * NV))
    beta0 = ttnn.reshape(ttnn.sigmoid(b), (1, 1, NV))
    g0 = ttnn.reshape(ttnn.multiply(neg_exp_a, _softplus_add(a, dt), dtype=f32), (1, 1, NV))
    o0, s0 = fused_recurrent_trim(q, k, v, beta0, g0, D**-0.5, initial_state=dev(s_t, f32))

    # fused: the fp32 a|b row and the constants row, straight into the op
    consts = ttnn.typecast(ttnn.concat([neg_exp_a, dt], dim=-1), f32)
    o1, s1 = fused_recurrent_trim(q, k, v, consts, ttnn.typecast(ab, f32), D**-0.5, initial_state=dev(s_t, f32),
                                  fused_gating=True)

    # float64 reference from the same bf16 inputs (q/k l2-normed, q scaled; GQA hv -> hv // 3)
    nea = ttnn.to_torch(neg_exp_a).double().reshape(NV)  # the same bf16 constants the device uses
    dtd = ttnn.to_torch(dt).double().reshape(NV)
    a64, b64 = ab_t.double().reshape(2 * NV)[:NV], ab_t.double().reshape(2 * NV)[NV:]
    decay = torch.exp(nea * torch.nn.functional.softplus(a64 + dtd))
    beta = torch.sigmoid(b64)
    qn = torch.nn.functional.normalize(q_t.double().reshape(NK, D), dim=-1) * D**-0.5
    kn = torch.nn.functional.normalize(k_t.double().reshape(NK, D), dim=-1)
    vv = v_t.double().reshape(NV, D)
    S = s_t.double() * decay[:, None, None]
    kh, qh = kn[torch.arange(NV) // 3], qn[torch.arange(NV) // 3]
    u = beta[:, None] * (vv - torch.einsum("hk,hkv->hv", kh, S))
    S = S + torch.einsum("hk,hv->hkv", kh, u)
    o_ref = torch.einsum("hk,hkv->hv", qh, S)

    for name, o, s in (("today's chain", o0, s0), ("fused gating", o1, s1)):
        o = ttnn.to_torch(o).double().reshape(NV, D)
        s = ttnn.to_torch(s).double().reshape(NV, D, D)
        eo = ((o - o_ref).abs().max() / o_ref.abs().max()).item()
        es = ((s - S).abs().max() / S.abs().max()).item()
        print(f"full step {name:<14}: o max rel err vs float64 {eo:.3e} (mean {(o - o_ref).abs().mean().item():.3e})"
              f"   state {es:.3e} (mean {(s - S).abs().mean().item():.3e})")
    assert torch.isfinite(ttnn.to_torch(s1)).all() and torch.isfinite(ttnn.to_torch(o1)).all()

    # Probes: effective per-head decay (k = 0: S_new = decay * S) and beta (S = 0: S_new = beta * kn (x) v).
    zero_k = dev(torch.zeros(1, 1, NK, D).to(bf))
    s_unit = torch.randn(NV, D, D)

    def run(beta_in, g_in, kk, s0_t, **kw):
        _, st = fused_recurrent_trim(q, kk, v, beta_in, g_in, D**-0.5, initial_state=dev(s0_t, f32), **kw)
        return ttnn.to_torch(st).double().reshape(NV, D, D)

    def fit(y, x):  # per-head least-squares scalar: y ~ c * x
        return (y * x).sum((1, 2)) / (x * x).sum((1, 2))

    kv = torch.einsum("hk,hv->hkv", kh, vv)
    results = {}
    for name, (bi, gi, kw) in {
        "today's chain": (beta0, g0, {}),
        "fused gating": (consts, ttnn.typecast(ab, f32), {"fused_gating": True}),
    }.items():
        d_eff = fit(run(bi, gi, zero_k, s_unit, **kw), s_unit.double())
        b_eff = fit(run(bi, gi, k, torch.zeros(NV, D, D), **kw), kv)
        ed, eb = ((d_eff - decay).abs() / decay).max().item(), ((b_eff - beta).abs() / beta).max().item()
        results[name] = (ed, eb)
        print(f"probe {name:<14}: decay max rel err vs float64 {ed:.3e}   beta {eb:.3e}")
    (ed0, eb0), (ed1, eb1) = results["today's chain"], results["fused gating"]
    assert ed1 <= ed0 * 1.05 and eb1 <= eb0 * 1.05, results
