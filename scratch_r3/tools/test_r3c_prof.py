# Round 3 matmul, third review of #58714 (2026-10-07): the bmm kernel's reload path (packer_l1_acc off: partials are
# reloaded into DEST with reload_from_dfb_to_dst and a matmul_block_init per sub block and k block) above LoFi at
# in0_block_w 1 (auto configs, DRAM interleaved), with a 16-bit and a 32-bit DEST, and Gemma-3-4B's vision QKV with
# packer L1 accumulation on and off, under the device profiler.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn

BF, B8, F32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32
FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi3": ttnn.MathFidelity.HiFi3, "HiFi4": ttnn.MathFidelity.HiFi4}

# (name, M, K, N, in1 dtype, out dtype, fidelity, fp32 DEST, packer_l1_acc)
CASES = [
    ("gemma3_vis_qkv_hifi4_fp32_l1acc", 4096, 1152, 4608, B8, BF, "HiFi4", True, True),
    ("gemma3_vis_qkv_hifi4_fp32_reload", 4096, 1152, 4608, B8, BF, "HiFi4", True, False),
    ("a2d_4k_1k_640_hifi2_fp32_reload", 4096, 1024, 640, B8, BF, "HiFi2", True, False),
    ("a2d_128_4k_512_hifi3_fp32_f32out_reload", 128, 4096, 512, B8, F32, "HiFi3", True, False),
    ("p2k_5k_5k_bfp8_hifi2_reload", 2048, 5120, 5120, B8, BF, "HiFi2", False, False),
    ("p4k_4k_4k_bf16_hifi2_reload", 4096, 4096, 4096, BF, BF, "HiFi2", False, False),
    ("p4k_1k_4k_bfp8_hifi4_reload", 4096, 1024, 4096, B8, BF, "HiFi4", False, False),
    ("p2k_8k_2k_bfp8_hifi2_fp32_reload", 2048, 8192, 2048, B8, BF, "HiFi2", True, False),
]


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_r3c_mm(device, case):
    name, m, k, n, d1, do, fid, fp32, l1acc = case
    torch.manual_seed(0)
    D = ttnn.DRAM_MEMORY_CONFIG
    a = ttnn.from_torch(torch.randn(1, 1, m, k) * 0.1, dtype=BF, layout=ttnn.TILE_LAYOUT, device=device, memory_config=D)
    b = ttnn.from_torch(torch.randn(1, 1, k, n) * 0.1, dtype=d1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=D)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=l1acc
    )
    out = ttnn.matmul(a, b, memory_config=D, dtype=do, compute_kernel_config=ckc)
    ttnn.synchronize_device(device)
    if _os.environ.get("V12_OUT"):
        t = ttnn.to_torch(out).contiguous()
        torch.save(t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32), _os.path.join(_os.environ["V12_OUT"], f"r3c_{name}.pt"))
    out.deallocate()
    a.deallocate()
    b.deallocate()
