# Round 3 matmul, third review of #58713 (2026-10-08): ttnn auto configs (DRAM interleaved, no program config) that compile
# to 2x3 sub blocks with a 16-bit DEST on the P100 (11 columns) and P150 (13 columns) grids, the block of the TTSync form's
# harness losses (2x3x4 half sync); under the device profiler.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn

BF, B8 = ttnn.bfloat16, ttnn.bfloat8_b
FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2}
# (name, M, K, N, in1 dtype, fidelity)
CASES = [
    ("a23_1280_4096_1056_bf16_lofi", 1280, 4096, 1056, BF, "LoFi"),
    ("a23_1280_4096_1248_bf16_lofi", 1280, 4096, 1248, BF, "LoFi"),
    ("a23_1280_4096_1056_bfp8_hifi2", 1280, 4096, 1056, B8, "HiFi2"),
    ("a23_1280_4096_1248_bfp8_hifi2", 1280, 4096, 1248, B8, "HiFi2"),
]


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_a23_mm(device, case):
    name, m, k, n, d1, fid = case
    torch.manual_seed(0)
    D = ttnn.DRAM_MEMORY_CONFIG
    a = ttnn.from_torch(torch.randn(1, 1, m, k) * 0.1, dtype=BF, layout=ttnn.TILE_LAYOUT, device=device, memory_config=D)
    b = ttnn.from_torch(torch.randn(1, 1, k, n) * 0.1, dtype=d1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=D)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    out = ttnn.matmul(a, b, memory_config=D, dtype=BF, compute_kernel_config=ckc)
    ttnn.synchronize_device(device)
    if _os.environ.get("V12_OUT"):
        torch.save(ttnn.to_torch(out).contiguous().view(torch.int16), _os.path.join(_os.environ["V12_OUT"], f"a23_{name}.pt"))
    out.deallocate()
    a.deallocate()
    b.deallocate()
