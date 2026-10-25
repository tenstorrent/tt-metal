import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_router_mm import PARAMS, chain_ms, make_gate, router_mm, up


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_prof(mesh_device):
    md = mesh_device
    gate, _ = make_gate(md, 2, 8)
    h = up(md, torch.randn(4, 5120))
    for S in (1, 8):
        wk, bt = gate._mm_consts(S)
        for mode in (3, 1, 0):
            if mode == 3 and S > 1:
                continue
            print(
                f"PROF S={S} mode={mode} {chain_ms(md, lambda: router_mm(h, wk, bt, S, mode=mode)) * 1e3:7.1f} us",
                flush=True,
            )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_prof_zones(mesh_device):
    md = mesh_device
    gate, _ = make_gate(md, 2, 8)
    h = up(md, torch.randn(4, 5120))
    S = int(os.environ.get("RMM_S1", "8"))
    wk, bt = gate._mm_consts(S)
    for _ in range(4):
        router_mm(h, wk, bt, S)
    ttnn.synchronize_device(md)
    ttnn.ReadDeviceProfiler(md)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_prof_sweep(mesh_device):
    md = mesh_device
    gate, _ = make_gate(md, 2, 8)
    h = up(md, torch.randn(4, 5120))
    for S in (1, 2, 4, 8, 10):
        wk, bt = gate._mm_consts(S)
        KS = 160 // S
        for B in sorted({b for b in (KS, KS // 2, 10, 5, 4, 2) if b >= 1 and KS % b == 0}, reverse=True):
            print(
                f"SWEEP S={S} B={B} {chain_ms(md, lambda: router_mm(h, wk, bt, S, block=B)) * 1e3:7.1f} us", flush=True
            )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_prof_skip(mesh_device):
    md = mesh_device
    gate, _ = make_gate(md, 2, 8)
    h = up(md, torch.randn(4, 5120))
    for S in (4,):
        wk, bt = gate._mm_consts(S)
        for skip in ("", "x", "w", "xw"):
            os.environ["RMM_SKIP"] = skip
            print(
                f"SKIP S={S} skip='{skip}' {chain_ms(md, lambda: router_mm(h, wk, bt, S, block=5)) * 1e3:7.1f} us",
                flush=True,
            )
