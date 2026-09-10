# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostics: (a) KDA prefill clamp level on the real layer-0 weights; (b) router precision (bf16 vs fp32 scores/topk)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.kda_ref import kda_layer_reference
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.moe_ref import router_reference
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tests.utils import first_shard, pcc, replicated
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.kda.layer import KimiKDA
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig

FULL = KDAConfig(hidden_size=2304, num_heads=32, head_k_dim=128, head_v_dim=128, conv_kernel_size=4, norm_eps=1e-5)


def test_kda_clamp_sweep(mesh_device, ccl, checkpoint, goldens):
    w = checkpoint.attention_state_dict(0)
    x_real = goldens["runs"]["prefill128"]["hooks"]["layer0.attn"]["in"]
    x_real = (
        (x_real if torch.is_tensor(x_real) else x_real[0]).float()[:, :128].bfloat16()
    )  # real post-norm activations [1,128,H]
    ref_out, ref_state = kda_layer_reference(x_real, w, FULL)
    layer = KimiKDA(mesh_device, FULL, w, layer_idx=0, ccl=ccl)
    for clamp in (-8.0, -6.0, -5.0, -4.0, -3.0, -2.0):
        layer.gate_clamp_min = clamp
        st = layer.allocate_prefill_state()
        out, ns = layer.forward_prefill(replicated(mesh_device, x_real), st, valid_len=128)
        o = first_shard(out).float()[0, 0]
        s = ttnn.to_torch(ns.recurrent).float()
        print(
            f"[clamp {clamp:5.1f}] real-activation prefill T=128: out pcc {pcc(ref_out[0], o):.5f} state pcc {pcc(ref_state.recurrent, s):.5f}"
        )


def test_router_precision(mesh_device, hf_config, checkpoint):
    sd = checkpoint.moe_state_dict(1, experts=False)
    Wg, bias = sd["moe.gate.weight"], sd["moe.gate.e_score_correction_bias"]
    torch.manual_seed(10)
    x = (torch.randn(1, 1, 32, hf_config.hidden_size) * 0.3).bfloat16()
    idx_ref, _, dense_ref = router_reference(x, Wg, bias, hf_config)
    chosen_ref = dense_ref > 0
    xt = replicated(mesh_device, x)
    W = replicated(mesh_device, Wg.T.reshape(1, 1, hf_config.hidden_size, -1).contiguous())
    b16 = replicated(mesh_device, bias.reshape(1, 1, 1, -1).bfloat16())
    b32 = replicated(mesh_device, bias.reshape(1, 1, 1, -1).float(), dtype=ttnn.float32)
    hifi4 = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
    )

    def agreement(choice, label):
        try:
            vals, idx = ttnn.topk(choice, k=8, dim=-1)
        except Exception as e:  # fp32 topk unsupported -> cast
            print(f"[router] topk on {choice.dtype} failed ({str(e)[:80]}); casting to bf16")
            vals, idx = ttnn.topk(ttnn.typecast(choice, ttnn.bfloat16), k=8, dim=-1)
        idx_t = ttnn.to_torch(idx).long()[0, 0]
        chosen = torch.zeros(32, 256, dtype=torch.bool).scatter(-1, idx_t, True)
        a = (chosen & chosen_ref).sum().item() / chosen_ref.sum().item()
        print(f"[router] {label}: top-8 agreement {a:.4f}")
        return a

    logits16 = ttnn.linear(xt, W, compute_kernel_config=hifi4)
    agreement(ttnn.add(ttnn.sigmoid(logits16), b16), "bf16 logits/sigmoid/bias")
    logits32 = ttnn.linear(xt, W, compute_kernel_config=hifi4, dtype=ttnn.float32)
    ch32 = ttnn.add(ttnn.sigmoid(logits32), b32)
    print("choice32 dtype", ch32.dtype)
    agreement(ch32, "fp32 logits/sigmoid/bias")
    # how close are the ties in the reference? (gap between 8th and 9th choice score)
    scores = torch.sigmoid(F.linear(x.float().reshape(-1, hf_config.hidden_size), Wg.float())) + bias.float()
    top9 = scores.topk(9, dim=-1).values
    print(
        f"[router] reference gap 8th-9th: median {(top9[:, 7] - top9[:, 8]).median():.5f}, min {(top9[:, 7] - top9[:, 8]).min():.6f}; bf16 ulp near 1.0 = 0.0078"
    )
    # fp32 matmul vs torch fp32 logits
    print(
        f"[router] fp32 device logits vs torch: pcc {pcc(F.linear(x.float().reshape(-1, hf_config.hidden_size), Wg.float()), ttnn.to_torch(logits32).float()[0, 0]):.6f}"
    )
