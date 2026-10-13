# #59149 regression audit: production kernels whose pack untilize takes the paced form (a 32-bit pack source in blocks
# of three) under the device profiler, to compare main, the head and the head without the pace (optin variant).
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)


MM = [
    # (name, M, K, N, grid x, out dtype)
    ("mm_untilize_out_fp32_n768", 128, 1024, 768, 8, "fp32"),
    ("mm_untilize_out_fp32_n288", 256, 512, 288, 3, "fp32"),
    ("mm_untilize_out_bf16_n768", 128, 1024, 768, 8, "bf16"),
]
DT = {"fp32": ttnn.float32, "bf16": ttnn.bfloat16}


@pytest.mark.parametrize("case", MM, ids=[c[0] for c in MM])
def test_mm_untilize_out(device, case):
    name, m, k, n, gx, od = case
    torch.manual_seed(0)
    a = torch.randn(1, 1, m, k)
    b = torch.randn(1, 1, k, n)
    ta = ttnn.from_torch(a, dtype=DT[od], layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=DT[od], layout=ttnn.TILE_LAYOUT, device=device)
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, 1),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=m // 32,
        per_core_N=n // 32 // gx,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
        untilize_out=True,
    )
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    out = ttnn.matmul(ta, tb, program_config=pc, compute_kernel_config=cfg, dtype=DT[od])
    got = ttnn.to_torch(out).float().reshape(m, n)
    ref = (a.reshape(m, k) @ b.reshape(k, n)).float()
    assert torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1] > 0.999


UT = [("untilize_fp32_w96", 2048, 96), ("untilize_fp32_w288", 1024, 288)]


@pytest.mark.parametrize("case", UT, ids=[c[0] for c in UT])
def test_untilize_fp32(device, case):
    name, h, w = case
    torch.manual_seed(0)
    x = torch.randn(1, 1, h, w)
    t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.untilize(t)
    assert torch.equal(ttnn.to_torch(out).reshape(h, w), x.reshape(h, w))
