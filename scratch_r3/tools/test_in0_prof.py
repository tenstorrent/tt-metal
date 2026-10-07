# Round 3 matmul, third review of #58712 (2026-10-07): ttnn auto-config matmuls whose sub block is taller than wide
# (rt > ct, so in0 is the streamed operand) with an 8-bit in0 and a 32-bit DEST, the case the 8-bit stream leaves at
# main's rate; under the device profiler. Shapes picked by their compiled sub blocks on the P100 and P150 mock clusters.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn

BF, B8 = ttnn.bfloat16, ttnn.bfloat8_b
FID = {"HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4, "LoFi": ttnn.MathFidelity.LoFi}

# (name, M, K, N, in0 dtype, fidelity, fp32 DEST)
CASES = [
    ("i0_512_4k_256_hifi2_fp32", 512, 4096, 256, B8, "HiFi2", True),
    ("i0_512_4k_256_hifi2_fp32_bf16in0", 512, 4096, 256, BF, "HiFi2", True),
    ("i0_2k_4k_352_hifi2_fp32", 2048, 4096, 352, B8, "HiFi2", True),
    ("i0_2k_4k_416_hifi2_fp32", 2048, 4096, 416, B8, "HiFi2", True),
    ("i0_4k_2k_352_hifi4_fp32", 4096, 2048, 352, B8, "HiFi4", True),
    ("i0_4k_2k_416_hifi4_fp32", 4096, 2048, 416, B8, "HiFi4", True),
    ("i0_1k_8k_224_hifi2_fp32", 1024, 8192, 224, B8, "HiFi2", True),
    ("i0_2k_2k_96_hifi2_fp32", 2048, 2048, 96, B8, "HiFi2", True),
    ("i0_512_4k_256_hifi2", 512, 4096, 256, B8, "HiFi2", False),
]


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_in0_mm(device, case):
    name, m, k, n, d0, fid, fp32 = case
    torch.manual_seed(0)
    D = ttnn.DRAM_MEMORY_CONFIG
    a = ttnn.from_torch(torch.randn(1, 1, m, k) * 0.1, dtype=d0, layout=ttnn.TILE_LAYOUT, device=device, memory_config=D)
    b = ttnn.from_torch(torch.randn(1, 1, k, n) * 0.1, dtype=B8, layout=ttnn.TILE_LAYOUT, device=device, memory_config=D)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=True
    )
    out = ttnn.matmul(a, b, memory_config=D, dtype=BF, compute_kernel_config=ckc)
    ttnn.synchronize_device(device)
    if _os.environ.get("V12_OUT"):
        torch.save(ttnn.to_torch(out).contiguous().view(torch.int16), _os.path.join(_os.environ["V12_OUT"], f"in0_{name}.pt"))
    out.deallocate()
    a.deallocate()
    b.deallocate()
