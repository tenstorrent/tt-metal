# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: compare every KDA prefill stage on device with the torch oracle (real layer-0 weights, T=32)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tests.utils import pcc, replicated
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.kda.layer import KimiKDA
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.ops import (
    causal_depthwise_conv_reference,
    kda_gate_reference,
    kda_recurrent_reference,
    sigmoid_gated_rms_norm_reference,
)
from models.demos.deepseek_v3_d_p.reference.kda.weights import normalize_kda_state_dict

FULL = KDAConfig(hidden_size=2304, num_heads=32, head_k_dim=128, head_v_dim=128, conv_kernel_size=4, norm_eps=1e-5)


def test_kda_stages(mesh_device, ccl, checkpoint):
    w = normalize_kda_state_dict(checkpoint.attention_state_dict(0), FULL)
    torch.manual_seed(2)
    T = 32
    hidden = (torch.randn(1, T, FULL.hidden_size) * 0.5).to(torch.bfloat16)
    x = hidden.float()
    # ---- torch stages
    q_r, _ = causal_depthwise_conv_reference(F.linear(x, w["q_proj.weight"].float()), w["q_conv1d.weight"])
    k_r, _ = causal_depthwise_conv_reference(F.linear(x, w["k_proj.weight"].float()), w["k_conv1d.weight"])
    v_r, _ = causal_depthwise_conv_reference(F.linear(x, w["v_proj.weight"].float()), w["v_conv1d.weight"])
    raw_gate = F.linear(F.linear(x, w["f_a_proj.weight"].float()), w["f_b_proj.weight"].float()).reshape(1, T, 32, 128)
    g_r = kda_gate_reference(raw_gate, w["A_log"], w["dt_bias"], None)  # [1,T,H,K]
    beta_r = torch.sigmoid(F.linear(x, w["b_proj.weight"].float()))  # [1,T,H]
    o_r, s_r = kda_recurrent_reference(
        q_r.reshape(1, T, 32, 128), k_r.reshape(1, T, 32, 128), v_r.reshape(1, T, 32, 128), g_r, beta_r
    )
    og_r = F.linear(F.linear(x, w["g_a_proj.weight"].float()), w["g_b_proj.weight"].float()).reshape(1, T, 32, 128)
    n_r = sigmoid_gated_rms_norm_reference(o_r, og_r, w["o_norm.weight"], FULL.norm_eps).reshape(1, T, 4096)
    out_r = F.linear(n_r, w["o_proj.weight"].float())
    print(
        f"[stats] g: min {g_r.min():.3f} mean {g_r.mean():.3f} max {g_r.max():.3f}; beta mean {beta_r.mean():.3f}; |q| {q_r.abs().mean():.3f} |v| {v_r.abs().mean():.3f}"
    )
    # ---- device stages
    layer = KimiKDA(mesh_device, FULL, checkpoint.attention_state_dict(0), layer_idx=0, ccl=ccl)
    kda = layer.kda
    st = layer.allocate_prefill_state()
    h_tt = replicated(mesh_device, hidden)
    p = kda._project_inputs(h_tt)
    q, k, v, _ = layer._convolve(p.qkv, st.convolution, T)
    gate, beta = kda._compute_gates(beta=p.beta, decay_rank=p.decay_rank)
    tt = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    print("shapes q", q.shape, "gate", gate.shape, "beta", beta.shape)
    print(
        f"[pcc] conv q {pcc(q_r, tt(q).reshape(q_r.shape)):.5f} k {pcc(k_r, tt(k).reshape(k_r.shape)):.5f} v {pcc(v_r, tt(v).reshape(v_r.shape)):.5f}"
    )
    g_tt = tt(gate).reshape(1, T, 32, 128)
    print(
        f"[pcc] gate {pcc(g_r, g_tt):.5f}  (tt gate min {g_tt.min():.3f} mean {g_tt.mean():.3f}; max abs diff {(g_r - g_tt).abs().max():.4f})"
    )
    print(f"[pcc] beta {pcc(beta_r, tt(beta).reshape(beta_r.shape)):.5f}")
    new_rec, out = kda.recurrence(q=q, k=k, v=v, gate=gate, beta=beta, initial_state=st.recurrent)
    print("recurrence out shape", out.shape, "state", new_rec.shape)
    o_tt = tt(out)  # [H, T, V]
    print(
        f"[pcc] recurrence out {pcc(o_r.permute(0, 2, 1, 3).reshape(32, T, 128), o_tt.reshape(32, T, 128)):.5f}  state {pcc(s_r, tt(new_rec)):.5f}"
    )
    n_tt = kda._kda_rms_norm(out, p.output_gate)
    print("norm out shape", n_tt.shape)
    print(f"[pcc] gated norm {pcc(n_r, tt(n_tt).reshape(n_r.shape)):.5f}")
    o = ttnn.linear(n_tt, layer.weights.output_projection, compute_kernel_config=kda.output_projection_compute_config)
    print(f"[pcc] final {pcc(out_r, tt(o).reshape(out_r.shape)):.5f}")
    # reference recurrence fed with the DEVICE q,k,v,gate,beta: isolates recurrence numerics from upstream
    o_x, s_x = kda_recurrent_reference(
        tt(q).reshape(1, T, 32, 128),
        tt(k).reshape(1, T, 32, 128),
        tt(v).reshape(1, T, 32, 128),
        g_tt,
        tt(beta).reshape(1, T, 32),
    )
    print(
        f"[pcc] torch recurrence on device inputs vs device recurrence: out {pcc(o_x.permute(0,2,1,3).reshape(32,T,128), o_tt.reshape(32,T,128)):.5f} state {pcc(s_x, tt(new_rec)):.5f}"
    )
    print(f"[pcc] torch recurrence on device inputs vs full torch: out {pcc(o_r, o_x):.5f}")
