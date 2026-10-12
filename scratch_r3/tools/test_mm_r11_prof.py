# Round 3 matmul r11: bfp8 matmuls with a 32-bit DEST (the kernels the r11 form returns to main's 8-bit body) under the device profiler (one launch per repetition), to compare main
# against the branch: prefill shapes whose sub blocks hold eight tiles (row MOP) and decode shapes with one-row and one-tile sub blocks
# (unpack k loop), bf16 and bfp8 inputs, LoFi to HiFi4, fp32 DEST on and off.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn

CASES = [
    # (name, M, K, N, in0 dtype, in1 dtype, fidelity, fp32 dest)
    ("prefill_2k_2k_8k_bfp8_lofi_fp32", 2048, 2048, 8192, "bf16", "bfp8", "LoFi", True),
    ("prefill_512_4k_4k_bfp8_hifi2_fp32", 512, 4096, 4096, "bf16", "bfp8", "HiFi2", True),
    ("decode_32_4k_14k_bfp8_hifi2_fp32", 32, 4096, 14336, "bf16", "bfp8", "HiFi2", True),
    ("decode_32_14k_1k_bfp8_lofi_fp32", 32, 14336, 1024, "bf16", "bfp8", "LoFi", True),
]
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}


@pytest.fixture(scope="module")
def device():
    from tests.tests_common.cache_entries_counter import CacheEntriesCounter

    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    dev.cache_entries_counter = CacheEntriesCounter(dev)
    yield dev
    ttnn.close_device(dev)
FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_mm_prof(device, case):
    name, m, k, n, d0, d1, fid, fp32 = case
    torch.manual_seed(0)
    a = ttnn.from_torch(torch.randn(1, 1, m, k) * 0.1, dtype=DT[d0], layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.randn(1, 1, k, n) * 0.1, dtype=DT[d1], layout=ttnn.TILE_LAYOUT, device=device)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=True
    )
    out = ttnn.matmul(a, b, compute_kernel_config=ckc)
    ttnn.synchronize_device(device)
    out.deallocate()
    a.deallocate()
    b.deallocate()
