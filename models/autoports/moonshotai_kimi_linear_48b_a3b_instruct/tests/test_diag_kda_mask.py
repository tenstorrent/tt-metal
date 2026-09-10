# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: does forcing fp32 outputs from prepare_chunk_recurrence (bf16 mask = 0) fix the fast-head state error?"""
from __future__ import annotations

import torch

import models.demos.deepseek_v3_d_p.tt.kda.recurrence as rec_mod
import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.kda_ref import kda_layer_reference
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tests.utils import first_shard, pcc, replicated
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.kda.layer import KimiKDA
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig

FULL = KDAConfig(hidden_size=2304, num_heads=32, head_k_dim=128, head_v_dim=128, conv_kernel_size=4, norm_eps=1e-5)


def test_mask_sweep(mesh_device, ccl, checkpoint, goldens):
    w = checkpoint.attention_state_dict(0)
    xr = goldens["runs"]["prefill128"]["hooks"]["layer0.attn"]["in"]
    x = (xr if torch.is_tensor(xr) else xr[0]).float().bfloat16()
    ref_out, ref_state = kda_layer_reference(x, w, FULL)
    layer = KimiKDA(mesh_device, FULL, w, layer_idx=0, ccl=ccl, gate_clamp_min=-5.0)
    default_mask = rec_mod.KDA_PREP_OUTPUT_BF16_MASK
    print(
        "[info] default KDA_PREP_OUTPUT_BF16_MASK =",
        bin(default_mask),
        "module names",
        [n for n in dir(rec_mod) if n.startswith("KDA_")],
    )
    for mask in (default_mask, 0):
        rec_mod.KDA_PREP_OUTPUT_BF16_MASK = mask
        try:
            st = layer.allocate_prefill_state()
            out, ns = layer.forward_prefill(replicated(mesh_device, x), st, valid_len=128)
            s = ttnn.to_torch(ns.recurrent).float()
            ph = torch.tensor([pcc(ref_state.recurrent[0, h], s[0, h]) for h in range(32)])
            print(
                f"[mask {bin(mask)}] out pcc {pcc(ref_out[0], first_shard(out).float()[0, 0]):.5f} state pcc {pcc(ref_state.recurrent, s):.5f} per-head min {ph.min():.3f} median {ph.median():.3f} dev|S|max {s.abs().max():.2f}"
            )
        except Exception as e:
            print(f"[mask {bin(mask)}] failed: {str(e)[:200]}")
    rec_mod.KDA_PREP_OUTPUT_BF16_MASK = default_mask
    # gate dtype experiment: feed the recurrence an fp32 gate instead of bf16 (KDA_GATE_DTYPE) if the op accepts it
    kda = layer.kda
    st = layer.allocate_prefill_state()
    p = kda._project_inputs(replicated(mesh_device, x))
    q, k, v, _ = layer._convolve(p.qkv, st.convolution, 128)
    gate, beta = kda._compute_gates(beta=p.beta, decay_rank=p.decay_rank)
    gate = ttnn.clamp(gate, min=-5.0, max=0.0)
    try:
        g32 = ttnn.typecast(gate, ttnn.float32)
        new_rec, out = kda.recurrence(q=q, k=k, v=v, gate=g32, beta=beta, initial_state=st.recurrent)
        s = ttnn.to_torch(new_rec).float()
        print(f"[fp32 gate] state pcc {pcc(ref_state.recurrent, s):.5f}")
    except Exception as e:
        print(f"[fp32 gate] rejected: {str(e)[:200]}")
