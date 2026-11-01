"""zone-profile driver of the fused expand + front (DSV41_MHC_LANEPAR): eager calls of DSV41MHC.expand_front (and, for comparison, expand + mixes_collapse) for
T in DSV41_MHC_PAR_T under the device profiler (TT_METAL_DEVICE_PROFILER=1); zones analysed by tests/mhc_zone_sum.py."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_mhc_lane import set_arm
from models.demos.blackhole.deepseek_v41_flash.tests.test_mhc_par import DEV, D, Ts, mk
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DEV, id="ring")], indirect=True)
@torch.no_grad()
def test_mhc_lane_prof(mesh_device):
    md = mesh_device
    up = mk(md)
    set_arm("base")
    m = DSV41MHC(md, torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5]))
    w = up(torch.rand(1, 1, 1, D) + 0.5)
    for T in Ts("4,16"):
        torch.manual_seed(T)
        x = up(torch.randn(T, 1, 4, D) + 0.3 * torch.randn(T, 1, 1, D))
        pre_in = up(torch.rand(T, 1, 1, 4))
        a = up(torch.randn(1, 1, T, D), ttnn.bfloat16)
        post, comb = up(torch.rand(T, 1, 4, 1)), up(torch.rand(T, 1, 4, 4))

        def lane():
            set_arm("lanefuse")
            return m.expand_front(a, x, post, comb, pre_in, w, 1e-20, pool=("lane", "attn"))

        def unfused():
            set_arm("lane")
            os.environ["DSV41_MHC_LANEPAR"] = "0"
            os.environ["DSV41_MHC_FRONT_FAST"] = "1"
            os.environ["DSV41_MHC_EXPAND_VALID"] = "1"
            x2 = m.expand(a, x, post, comb, None, pool=("ref", "attn"))
            m.mixes_collapse(x2, pre_in, w, 1e-20, emit_rm=True)

        lane()
        unfused()
        ttnn.synchronize_device(md)
        ttnn.ReadDeviceProfiler(md)  # drop the warm-up
        for _ in range(2):
            lane()
            ttnn.synchronize_device(md)
            unfused()
            ttnn.synchronize_device(md)
        ttnn.ReadDeviceProfiler(md)
        print(f"PROF_DONE T={T}", flush=True)
