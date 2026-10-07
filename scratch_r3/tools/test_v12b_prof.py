# Round 3 matmul, second review of #58712 (2026-10-07): matmuls whose ttnn auto config (DRAM interleaved, no program
# config, the p100a's 11x10 grid) streams an 8-bit in1 at in0_block_w 1 in 1x2 sub blocks, one k step per DEST section,
# half sync, with a 16-bit and a 32-bit DEST (the configuration of the LLK perf gate's 1x2x1 Bfp8_b k 1 cells), a 2x1
# control (bf16 in0 streamed), Gemma-3-4B's vision QKV (auto config, HiFi4, fp32 DEST, bfp8 weights: 1x2 on the p100a),
# and the bfp8 prefills of the earlier lists, under the device profiler.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn

BF, B8, F32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32
FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi3": ttnn.MathFidelity.HiFi3, "HiFi4": ttnn.MathFidelity.HiFi4}

# (name, M, K, N, out dtype, fidelity, fp32 DEST)
CASES = [
    ("a2d_128_4k_512_hifi2", 128, 4096, 512, BF, "HiFi2", False),
    ("a2d_128_4k_512_hifi2_fp32", 128, 4096, 512, BF, "HiFi2", True),
    ("a2d_128_4k_512_hifi3_fp32_f32out", 128, 4096, 512, F32, "HiFi3", True),
    ("a2d_4k_1k_640_hifi2", 4096, 1024, 640, BF, "HiFi2", False),
    ("a2d_4k_1k_640_hifi2_fp32", 4096, 1024, 640, BF, "HiFi2", True),
    ("a2d_2k_4k_640_hifi2_fp32", 2048, 4096, 640, BF, "HiFi2", True),
    ("a1d_32_3424_5120_hifi2", 32, 3424, 5120, BF, "HiFi2", False),
    ("a1d_32_3424_5120_hifi2_fp32", 32, 3424, 5120, BF, "HiFi2", True),
    ("ctl_2x1_512_4k_256_hifi2", 512, 4096, 256, BF, "HiFi2", False),
    ("gemma3_vis_qkv_4k_1152_4608_hifi4_fp32", 4096, 1152, 4608, BF, "HiFi4", True),
    ("prefill_2k_2k_8k_bfp8_lofi", 2048, 2048, 8192, BF, "LoFi", False),
    ("prefill_2k_2k_8k_bfp8_lofi_fp32", 2048, 2048, 8192, BF, "LoFi", True),
]


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_v12b_mm(device, case):
    name, m, k, n, do, fid, fp32 = case
    torch.manual_seed(0)
    D = ttnn.DRAM_MEMORY_CONFIG
    a = ttnn.from_torch(torch.randn(1, 1, m, k) * 0.1, dtype=BF, layout=ttnn.TILE_LAYOUT, device=device, memory_config=D)
    b = ttnn.from_torch(torch.randn(1, 1, k, n) * 0.1, dtype=B8, layout=ttnn.TILE_LAYOUT, device=device, memory_config=D)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=True
    )
    out = ttnn.matmul(a, b, memory_config=D, dtype=do, compute_kernel_config=ckc)
    ttnn.synchronize_device(device)
    out.deallocate()
    a.deallocate()
    b.deallocate()
