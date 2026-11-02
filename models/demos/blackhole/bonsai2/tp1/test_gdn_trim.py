# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_1C_GDN_TRIM equivalence: fused_recurrent_trim against fused_recurrent_gated_delta_rule_ttnn
(high_precision) on decode-shaped inputs, g given as bf16 to the old path and fp32 to the new one, as the
producers hand it over. The folded q operand differs from the old one by 6e-8 relative, but the op rounds q
internally at about bf16 precision, so o moves by up to ~3.5e-4 relative (measured, scratch/trim_diag2.py:
the same 6e-8 change made in Python gives the same 3.5e-4). The tolerance is set at that floor; agreement
is decided by the e2e gate.

    pytest models/demos/blackhole/bonsai2/tp1/test_gdn_trim.py -s
"""
import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.tp import fused_recurrent_trim
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    fused_recurrent_gated_delta_rule_ttnn,
)

NK, NV, D = 16, 48, 128


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_gdn_trim(mesh_device):
    torch.manual_seed(0)

    def dev(t, dt=ttnn.bfloat16):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                               memory_config=ttnn.L1_MEMORY_CONFIG)

    q, k, v = (dev(torch.randn(1, 1, n, D)) for n in (NK, NK, NV))
    beta = dev(torch.rand(1, 1, NV))
    g_t = -torch.rand(1, 1, NV) * 2
    s_t = torch.randn(NV, D, D) * 0.1
    o0, s0 = fused_recurrent_gated_delta_rule_ttnn(q, k, v, beta, dev(g_t), scale=D**-0.5,
                                                   initial_state=dev(s_t, ttnn.float32), device=mesh_device)
    o1, s1 = fused_recurrent_trim(q, k, v, beta, dev(g_t.bfloat16().float(), ttnn.float32), D**-0.5,
                                  initial_state=dev(s_t, ttnn.float32))
    o2, s2 = fused_recurrent_gated_delta_rule_ttnn(q, k, v, beta, dev(g_t), scale=D**-0.5,
                                                   initial_state=dev(s_t, ttnn.float32), device=mesh_device)
    a, b = ttnn.to_torch(o0).float(), ttnn.to_torch(o2).float()
    print(f"old vs old rerun: o max rel {((a - b).abs().max() / a.abs().max()).item():.3e}")
    o3, s3 = fused_recurrent_gated_delta_rule_ttnn(q, k, v, beta, dev(g_t.bfloat16().float(), ttnn.float32),
                                                   scale=D**-0.5, initial_state=dev(s_t, ttnn.float32), device=mesh_device)
    b = ttnn.to_torch(o3).float()
    print(f"old with fp32 g vs old: o max rel {((a - b).abs().max() / a.abs().max()).item():.3e}")
    for name, a, b in (("o", o0, o1), ("state", s0, s1)):
        a, b = ttnn.to_torch(a).float(), ttnn.to_torch(b).float()
        rel = ((a - b).abs().max() / a.abs().max()).item()
        print(f"{name}: shape {tuple(b.shape)} bit-exact {100 * (a == b).float().mean().item():.2f}%  max rel diff {rel:.3e}")
        assert rel < 1e-3
