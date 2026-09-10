# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: why is the KDA prefill's final recurrent state inaccurate on real activations while outputs are fine?"""
from __future__ import annotations

import torch

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.kda_ref import kda_layer_reference
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tests.utils import first_shard, pcc, replicated
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.kda.layer import KimiKDA
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tt.kda.config import KDARecurrenceProgramConfig

FULL = KDAConfig(hidden_size=2304, num_heads=32, head_k_dim=128, head_v_dim=128, conv_kernel_size=4, norm_eps=1e-5)


def _per_head(a, b):
    return torch.tensor([pcc(a[0, h], b[0, h]) for h in range(a.shape[1])])


def test_state_diag(mesh_device, ccl, checkpoint, goldens):
    w = checkpoint.attention_state_dict(0)
    xr = goldens["runs"]["prefill128"]["hooks"]["layer0.attn"]["in"]
    xr = (xr if torch.is_tensor(xr) else xr[0]).float()
    xd = goldens["runs"]["decode128"]["hooks"]["layer0.attn"]["in"]
    xd = (xd if torch.is_tensor(xd) else xd[0]).float()  # [1,1,H] token 128
    x_all = torch.cat([xr, xd], dim=1).bfloat16()  # 129 tokens
    a_log = w["A_log"].float().reshape(-1)
    print("[info] exp(A_log) per head:", [round(v, 1) for v in a_log.exp().tolist()])
    layer = KimiKDA(mesh_device, FULL, w, layer_idx=0, ccl=ccl, gate_clamp_min=-5.0)
    for T in (32, 64, 128):
        ref_out, ref_state = kda_layer_reference(x_all[:, :T], w, FULL)
        st = layer.allocate_prefill_state()
        out, ns = layer.forward_prefill(replicated(mesh_device, x_all[:, :T]), st, valid_len=T)
        s = ttnn.to_torch(ns.recurrent).float()
        ph = _per_head(ref_state.recurrent, s)
        print(
            f"[T={T:3}] out pcc {pcc(ref_out[0], first_shard(out).float()[0, 0]):.5f} state pcc {pcc(ref_state.recurrent, s):.5f} "
            f"per-head state pcc min {ph.min():.3f} median {ph.median():.3f}; worst heads {ph.argsort()[:4].tolist()}; "
            f"ref |S| max {ref_state.recurrent.abs().max():.3f} dev |S| max {s.abs().max():.3f} finite {torch.isfinite(s).all().item()}"
        )
        if T == 128:
            # what matters: the next decode token computed from the device-carried state
            ds = layer.allocate_decode_state(batch=1)
            layer.prefill_state_to_decode(ns, ds)
            ref129, _ = kda_layer_reference(x_all, w, FULL)
            o = layer.forward_decode(replicated(mesh_device, x_all[:, 128:129].reshape(1, 1, 1, -1)), ds)
            print(
                f"[decode@128 from device prefill state] pcc {pcc(ref129[0, 128], first_shard(o).float().reshape(-1)):.5f}"
            )
            # same decode step but from the REFERENCE state uploaded to device (isolates the state error)
            ds2 = layer.allocate_decode_state(batch=1)
            ref_st = kda_layer_reference(x_all[:, :128], w, FULL)[1]
            ttnn.copy(
                ttnn.from_torch(ref_st.recurrent, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device),
                ds2.recurrent,
            )
            conv = torch.cat([ref_st.q_convolution, ref_st.k_convolution, ref_st.v_convolution], -1)  # [1,3,C]
            for j in range(3):
                ttnn.copy(
                    ttnn.from_torch(
                        conv[:, j : j + 1].bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
                    ),
                    ds2.conv_history[j],
                )
            o2 = layer.forward_decode(replicated(mesh_device, x_all[:, 128:129].reshape(1, 1, 1, -1)), ds2)
            print(
                f"[decode@128 from reference state] pcc {pcc(ref129[0, 128], first_shard(o2).float().reshape(-1)):.5f}"
            )
    # grouped scan variant
    layer.kda.recurrence = type(layer.kda.recurrence)(
        mesh_device,
        KDARecurrenceProgramConfig(local_scan_strategy="grouped", summary_group_chunks=4),
        sequence_parallel_axis=None,
    )
    ref_out, ref_state = kda_layer_reference(x_all[:, :128], w, FULL)
    st = layer.allocate_prefill_state()
    out, ns = layer.forward_prefill(replicated(mesh_device, x_all[:, :128]), st, valid_len=128)
    s = ttnn.to_torch(ns.recurrent).float()
    print(
        f"[grouped scan T=128] out pcc {pcc(ref_out[0], first_shard(out).float()[0, 0]):.5f} state pcc {pcc(ref_state.recurrent, s):.5f}"
    )
