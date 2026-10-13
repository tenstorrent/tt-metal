# Regression audit #58817: the 8-bit stream now runs only with a full-sync 16-bit DEST. bfp8 in1 streamed, bmm kernel at
# full sync, explicit 2D configs on an 8x8 grid (rows of 4 to 16 tiles), LoFi and HiFi2, plus an auto-config decode and
# prefill case at full sync. Device profiler, one launch per repetition (mm_prof_plugin).
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn

CASES = []
for fid in ("LoFi", "HiFi2"):
    for h, w in ((1, 4), (1, 8), (2, 4), (1, 12), (1, 16), (2, 8)):
        for ibw in (1, 2):
            CASES.append((f"sb{h}x{w}_ibw{ibw}_{fid}", h, w, ibw, fid))
FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2}


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


def ckc(device, fid):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True,
        dst_full_sync_en=True,
    )


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_fs_prof(device, case):
    name, h, w, ibw, fid = case
    gx, gy = 8, 8
    pcm, pcn = 2 * h, w if w > 8 else 2 * w
    m, k, n = gx * pcm * 32, 2048, gy * pcn * 32
    torch.manual_seed(0)
    a = ttnn.from_torch(torch.randn(1, 1, m, k) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.randn(1, 1, k, n) * 0.1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw, out_subblock_h=h, out_subblock_w=w,
        per_core_M=pcm, per_core_N=pcn, transpose_mcast=False, fused_activation=None,
    )
    out = ttnn.matmul(a, b, program_config=pc, compute_kernel_config=ckc(device, fid))
    ttnn.synchronize_device(device)
    out.deallocate()
    a.deallocate()
    b.deallocate()


@pytest.mark.parametrize("mkn", [(32, 8192, 4096), (2048, 2048, 8192)], ids=["decode_32x8192x4096", "prefill_2048x2048x8192"])
def test_fs_auto(device, mkn):
    m, k, n = mkn
    a = ttnn.from_torch(torch.randn(1, 1, m, k) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.randn(1, 1, k, n) * 0.1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.matmul(a, b, compute_kernel_config=ckc(device, "LoFi"))
    ttnn.synchronize_device(device)
    out.deallocate()
    a.deallocate()
    b.deallocate()
