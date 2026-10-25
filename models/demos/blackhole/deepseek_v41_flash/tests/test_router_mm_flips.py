# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Expert-selection flips of the split-K router (S > 1: different fp32 summation order) vs the ttnn chain on REAL FFN inputs
(layers 1/2/8/14/20) plus many small perturbations of them (real-like top-6/7 margins): count rows whose expert id list or
set differs, and report the closest 6th/7th rank margin seen."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_router_mm import (
    CHAIN,
    PARAMS,
    d0,
    make_gate,
    old_chain,
    rows,
    up,
)
from models.demos.blackhole.deepseek_v41_flash.tt.router_mm import router_fused
from models.demos.blackhole.deepseek_v41_flash.tt.router_select import router_select

SS = [int(v) for v in os.environ.get("RMM_S", "2,4").split(",")]
NVAR = int(os.environ.get("RMM_NVAR", "100"))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_router_mm_flips(mesh_device):
    md = mesh_device
    torch.manual_seed(5)
    for S in SS:
        tot = {"rows": 0, "ids": 0, "sets": 0}
        minmargin = 1e9
        for L in (1, 2, 8, 14, 20):
            x0 = torch.load(os.path.join(CHAIN, f"ffn_inputs_{L}.pt"))["x"].reshape(16, 5120).float()
            gate, _ = make_gate(md, L, 16)
            wk, bt = gate._mm_consts(S)
            for v in range(NVAR + 1):
                x = x0 if v == 0 else x0 * (1 + 0.02 * torch.randn(16, 5120)) + 0.02 * x0.std() * torch.randn(16, 5120)
                for T in (4, 8, 16):
                    h = up(md, x[:T])
                    ro, so = old_chain(gate, h)
                    wo, io = router_select(ro, so, gate.k, 1e-20, gate.scaling_factor)
                    wn, i_n = router_fused(h, wk, bt, S, gate.k, 1e-20, gate.scaling_factor)
                    a, b = d0(io, T).long(), d0(i_n, T).long()
                    tot["rows"] += T
                    tot["ids"] += int((a != b).any(-1).sum())
                    tot["sets"] += sum(set(p.tolist()) != set(q.tolist()) for p, q in zip(a, b))
                    top = rows(ro, T).topk(7, -1).values
                    minmargin = min(minmargin, float((top[:, 5] - top[:, 6]).min()))
                    for t in (h, ro, so, wo, io, wn, i_n):
                        ttnn.deallocate(t)
            print(f"FLIPS S={S} layer {L}: cumulative {tot}, closest 6th/7th margin {minmargin:.3e}", flush=True)
        print(f"FLIPS S={S} TOTAL {tot}", flush=True)
        assert tot["sets"] == 0
