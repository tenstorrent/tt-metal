# Round 3 matmul (#58714): prefill-shaped matmuls on ttnn's auto config (DRAM interleaved, no program config), which on
# the p100a's 11x10 and the P150's 13x10 grid mostly takes in0_block_w 1 and, with a 16-bit DEST, sub blocks of 8 tiles
# (4x2, 2x4, 1x8, 8x1): the shapes a row MOP gate of 8 tiles at any in0_block_w would add. Under the device profiler.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn

BF, B8, B4 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b
FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}
# (name, M, K, N, in0 dtype, in1 dtype, out dtype, fidelity)
CASES = [
    ("p2k_4k_4k_bfp8_lofi", 2048, 4096, 4096, BF, B8, BF, "LoFi"),
    ("p2k_4k_14k_bfp4_lofi", 2048, 4096, 14336, BF, B4, B8, "LoFi"),
    ("p2k_14k_4k_bfp8_lofi", 2048, 14336, 4096, B8, B8, BF, "LoFi"),
    ("p2k_2k_8k_bfp8_lofi", 2048, 2048, 8192, BF, B8, BF, "LoFi"),
    ("p4k_4k_1k_bf16_hifi2", 4096, 4096, 1024, BF, BF, BF, "HiFi2"),
    ("p1k_2k_2k_bf16_hifi2", 1024, 2048, 2048, BF, BF, BF, "HiFi2"),
    ("p2k_5k_5k_bfp8_hifi2", 2048, 5120, 5120, BF, B8, BF, "HiFi2"),
    ("p8k_1k_1k_bf16_hifi4", 8192, 1024, 1024, BF, BF, BF, "HiFi4"),
    ("p512_4k_4k_bfp8_hifi2", 512, 4096, 4096, BF, B8, BF, "HiFi2"),
    ("p4k_2k_2k_bfp8_lofi", 4096, 2048, 2048, B8, B8, B8, "LoFi"),
    ("p4k_4k_4k_bf16_hifi2", 4096, 4096, 4096, BF, BF, BF, "HiFi2"),
    ("p2k_8k_2k_bfp8_hifi2", 2048, 8192, 2048, BF, B8, BF, "HiFi2"),
    ("p1k_4k_11k_bf16_hifi4", 1024, 4096, 11008, BF, BF, BF, "HiFi4"),
    ("p2k_3k_3k_bfp8_hifi2", 2048, 3072, 3072, BF, B8, BF, "HiFi2"),
    ("p4k_1k_4k_bfp8_hifi4", 4096, 1024, 4096, BF, B8, BF, "HiFi4"),
    ("p2k_5k_5k_bfp8_lofi", 2048, 5120, 5120, BF, B8, BF, "LoFi"),
]


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_rk1_mm(device, case):
    name, m, k, n, d0, d1, do, fid = case
    torch.manual_seed(0)
    D = ttnn.DRAM_MEMORY_CONFIG
    a = ttnn.from_torch(torch.randn(1, 1, m, k) * 0.1, dtype=d0, layout=ttnn.TILE_LAYOUT, device=device, memory_config=D)
    b = ttnn.from_torch(torch.randn(1, 1, k, n) * 0.1, dtype=d1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=D)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    out = ttnn.matmul(a, b, memory_config=D, dtype=do, compute_kernel_config=ckc)
    ttnn.synchronize_device(device)
    out.deallocate()
    a.deallocate()
    b.deallocate()
