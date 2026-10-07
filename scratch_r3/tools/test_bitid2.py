# Round 3 matmul (07 evidence rule 3), the bmm kernel's row MOP above LoFi for every sub block and k: every bf16 bit pattern
# (65536, NaN, inf, denormals and signed zero included) as in0 or as in1 against a random other operand, on one core with
# the sub blocks the auto configs give (7x1, 2x3, 1x2 and 4x2 at in0_block_w 1, 1x1 at in0_block_w 8, 2x2 with a 32-bit
# DEST), at HiFi2, HiFi3 and HiFi4, and a LoFi control. Outputs saved as raw bits (V12_OUT) for a comparison across trees.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    _sys.exit("not under hwlock")
import os

import pytest
import torch
import ttnn

FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi3": ttnn.MathFidelity.HiFi3, "HiFi4": ttnn.MathFidelity.HiFi4}


def all_bf16(rows, cols, seed):
    # every bf16 pattern once, the rest of the tensor random
    g = torch.Generator().manual_seed(seed)
    t = (torch.randn(rows * cols, generator=g) * 0.5).to(torch.bfloat16).view(torch.int16)
    t[:65536] = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    return t.view(torch.bfloat16).reshape(rows, cols)


def rnd(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, cols, generator=g) * 0.5).to(torch.bfloat16)


# (name, M, K, N, which operand holds every pattern, sub h, sub w, in0_block_w, fidelity, fp32 DEST)
CASES = [
    ("s7x1_k1_hifi2_A", 224, 320, 32, "A", 7, 1, 1, "HiFi2", False),
    ("s7x1_k1_hifi2_B", 224, 256, 256, "B", 7, 1, 1, "HiFi2", False),
    ("s2x3_k1_hifi4_A", 64, 1024, 96, "A", 2, 3, 1, "HiFi4", False),
    ("s1x2_k1_hifi3_A", 32, 2048, 64, "A", 1, 2, 1, "HiFi3", False),
    ("s1x2_k1_hifi3_B", 32, 256, 256, "B", 1, 2, 1, "HiFi3", False),
    ("s1x1_k8_hifi2_A", 32, 2048, 32, "A", 1, 1, 8, "HiFi2", False),
    ("s4x2_k1_hifi2_A", 128, 512, 64, "A", 4, 2, 1, "HiFi2", False),
    ("s2x2_k2_hifi4_fp32_A", 64, 1024, 64, "A", 2, 2, 2, "HiFi4", True),
    ("s7x1_k1_lofi_A", 224, 320, 32, "A", 7, 1, 1, "LoFi", False),
]


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_bitid2(device, case):
    name, m, k, n, which, sh, sw, ibw, fid, fp32 = case
    a = all_bf16(m, k, 1) if which == "A" else rnd(m, k, 1)
    b = all_bf16(k, n, 2) if which == "B" else rnd(k, n, 2)
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False
    )
    cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(1, 1), in0_block_w=ibw, out_subblock_h=sh, out_subblock_w=sw,
        out_block_h=m // 32, out_block_w=n // 32, per_core_M=m // 32, per_core_N=n // 32, fuse_batch=True,
        fused_activation=None, mcast_in0=True,
    )
    out = ttnn.matmul(ta, tb, program_config=cfg, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
    got = ttnn.to_torch(out).reshape(m, n).contiguous().view(torch.int16)
    if os.environ.get("V12_OUT"):
        torch.save(got, os.path.join(os.environ["V12_OUT"], f"bitid2_{name}.pt"))
    assert got.shape == (m, n)
