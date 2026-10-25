# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""router_fused micro-benchmarks: trace timing sweeps (S, share, half, out_l1) and a device-profiler zone run
(``RMM_PROF=1`` with TT_METAL_DEVICE_PROFILER=1)."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_router_mm import PARAMS, chain_ms, make_gate, up
from models.demos.blackhole.deepseek_v41_flash.tt.router_mm import router_fused


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_prof(mesh_device):
    md = mesh_device
    gate, _ = make_gate(md, 2, 8)
    T = int(os.environ.get("RMM_T", "4"))
    h = up(md, torch.randn(T, 5120))
    S0 = int(os.environ.get("RMM_S1", "2"))
    if os.environ.get("RMM_PROF") == "1":
        wk, bt = gate._mm_consts(S0)
        for _ in range(4):
            router_fused(h, wk, bt, S0, 6, 1e-20, 1.5, out_l1=True)
        ttnn.synchronize_device(md)
        ttnn.ReadDeviceProfiler(md)
        return
    for S in (1, 2):
        wk, bt = gate._mm_consts(S)
        for base in ((0, 0), (0, 2), (0, 4), (0, 6), (0, 8)):
            if base[1] + S > 10:
                continue
            f = lambda: router_fused(h, wk, bt, S, 6, 1e-20, 1.5, out_l1=True, base=base)
            print(f"PROF T={T} S={S} base={base}: {chain_ms(md, f) * 1e3:7.1f} us", flush=True)
