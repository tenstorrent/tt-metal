# Regression audit #58817: the 8-bit stream (bfp8 in1 streamed, 16-bit DEST, half sync) at the bmm kernel's sub blocks of
# 3 to 8 tiles, explicit 2D configs on an 8x8 grid, LoFi (tile MOP below 6 wide) and HiFi2 (row MOP), plus the auto
# config prefill case of the PR body. Device profiler, one launch per repetition (mm_prof_plugin).
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn

CASES = []
for fid in ("LoFi", "HiFi2"):
    for w in (3, 5, 6, 7, 8):
        for ibw in (1, 2):
            CASES.append((f"sb1x{w}_ibw{ibw}_{fid}", w, ibw, fid))
FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2}


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_nar_prof(device, case):
    name, w, ibw, fid = case
    gx, gy, pcm = 8, 8, 4
    pcn = 2 * w
    m, k, n = gx * pcm * 32, 2048, gy * pcn * 32
    torch.manual_seed(0)
    a = ttnn.from_torch(torch.randn(1, 1, m, k) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.randn(1, 1, k, n) * 0.1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw, out_subblock_h=1, out_subblock_w=w,
        per_core_M=pcm, per_core_N=pcn, transpose_mcast=False, fused_activation=None,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    out = ttnn.matmul(a, b, program_config=pc, compute_kernel_config=ckc)
    ttnn.synchronize_device(device)
    if _os.environ.get("V12_OUT"):
        torch.save(ttnn.to_torch(out).contiguous().view(torch.int16), _os.path.join(_os.environ["V12_OUT"], f"nar_{name}.pt"))
    out.deallocate()
    a.deallocate()
    b.deallocate()


def test_nar_auto_prefill(device):
    a = ttnn.from_torch(torch.randn(1, 1, 2048, 2048) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.randn(1, 1, 2048, 8192) * 0.1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    out = ttnn.matmul(a, b, compute_kernel_config=ckc)
    ttnn.synchronize_device(device)
    out.deallocate()
    a.deallocate()
    b.deallocate()
