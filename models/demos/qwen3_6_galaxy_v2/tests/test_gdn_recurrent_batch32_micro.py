# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""GDN DeltaNet recurrent at B=32: does the batched recurrent isolate 32 independent user states?

Decode batch-32 fills the M=32 tile with 32 real users. The qwen3.6-unique risk is the GDN recurrent:
it must update 32 INDEPENDENT recurrent states (one per user) with no cross-contamination. This runs the
real recurrent_gated_delta_rule_ttnn_fp32 once at B=32 (T=1 decode step) with 32 distinct inputs +
distinct initial states, then re-runs each sampled user ALONE at B=1, and checks per-user PCC of both
output and final_state. Match => batched recurrent is correct for batch-32.

Run:
    python -m pytest --noconftest models/demos/qwen3_6_galaxy_v2/tests/test_gdn_recurrent_batch32_micro.py -v -s
"""
from __future__ import annotations

import os

import pytest
import torch

import ttnn

_T = 1  # decode step
_H = 6  # v-heads per chip (n_v_heads 48 -> 6/chip)
_K = 128  # head_dim
_V = 128
_SCALE = _K**-0.5
# Brackets the two grid-guard thresholds (outer-product B*H*4>120 -> B>5; read B*H>120 -> B>20)
# plus the production B=1. GDN_B env overrides to a single batch for debugging.
_B_SWEEP = [int(os.environ["GDN_B"])] if os.environ.get("GDN_B") else [1, 4, 5, 6, 8, 20, 21, 32]


@pytest.fixture(scope="module")
def dev():
    d = ttnn.open_device(device_id=0)
    yield d
    ttnn.close_device(d)


def _up(t, device):
    return ttnn.from_torch(
        t, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


@pytest.mark.hardware
@pytest.mark.parametrize("B", _B_SWEEP)
def test_gdn_recurrent_batchN(dev, B):
    """Batched recurrent must isolate B independent user states (grid-guard fix). The pre-fix
    MatmulMultiCoreReuseProgramConfig corrupted batch positions once B*H*M_blocks>120 (B>=8 outer,
    B>20 read); the grid guard falls back to auto-matmul above the grid. Per-user output+state PCC
    of the B=N batched run vs N standalone B=1 runs must stay >0.999."""
    from models.common.utility_functions import comp_pcc
    from models.demos.qwen3_6_galaxy_v2.tt.ttnn_delta_rule_ops_fp32 import recurrent_gated_delta_rule_ttnn_fp32

    torch.manual_seed(0)
    q = torch.randn(B, _T, _H, _K, dtype=torch.float32) * 0.1
    k = torch.randn(B, _T, _H, _K, dtype=torch.float32) * 0.1
    v = torch.randn(B, _T, _H, _V, dtype=torch.float32) * 0.1
    beta = torch.rand(B, _T, _H, dtype=torch.float32)  # (0,1)
    g = -torch.rand(B, _T, _H, dtype=torch.float32)  # decay (log-space, <0)
    S0 = torch.randn(B, _H, _K, _V, dtype=torch.float32) * 0.1  # distinct initial state/user

    def run(bsel):
        sl = slice(None) if bsel is None else slice(bsel, bsel + 1)
        out, st = recurrent_gated_delta_rule_ttnn_fp32(
            _up(q[sl], dev),
            _up(k[sl], dev),
            _up(v[sl], dev),
            _up(beta[sl], dev),
            _up(g[sl], dev),
            scale=_SCALE,
            initial_state=_up(S0[sl], dev),
            device=dev,
        )
        return ttnn.to_torch(out).float(), ttnn.to_torch(st).float()

    outN, stN = run(None)
    check_users = sorted(set([0, 1, 2, 3, min(7, B - 1), B // 2, B - 1]))
    check_users = [u for u in check_users if 0 <= u < B]
    print(f"\n  B={B} run: out={tuple(outN.shape)} state={tuple(stN.shape)}; per-user PCC vs standalone B=1:")
    worst_out, worst_st = 1.0, 1.0
    for u in check_users:
        out1, st1 = run(u)
        ok_o, m_o = comp_pcc(outN[u : u + 1], out1, 0.999)
        ok_s, m_s = comp_pcc(stN[u : u + 1], st1, 0.999)
        mo = float(str(m_o).split()[-1]) if not isinstance(m_o, float) else float(m_o)
        ms = float(str(m_s).split()[-1]) if not isinstance(m_s, float) else float(m_s)
        worst_out, worst_st = min(worst_out, mo), min(worst_st, ms)
        print(f"    user {u:2d}: output PCC={m_o}  state PCC={m_s}  {'OK' if (ok_o and ok_s) else 'FAIL'}")
    print(f"  B={B} worst-case: output PCC={worst_out}  state PCC={worst_st}")
    assert worst_out > 0.999 and worst_st > 0.999, f"B={B}: batched recurrent does NOT isolate users (batch bug)"
