# Scratch, not for merge: the production callers of REDUCE_COL SUM/AVG (generic reduce over H, moreh sum, mean and softmax over H).
# Each test runs the op once; the output goes through ttnn.to_torch so the arms' results can be hashed.
import pytest
import torch
import ttnn

SHAPES = {
    "1x1x1024x1024": (1, 1, 1024, 1024),
    "1x1x4096x128": (1, 1, 4096, 128),
    "4x1x2048x256": (4, 1, 2048, 256),
    "1x1x32x8192": (1, 1, 32, 8192),
    "1x1x256x2048": (1, 1, 256, 2048),
    "1x1x128x128": (1, 1, 128, 128),
}
DT = {"bf16": ttnn.bfloat16, "f32": ttnn.float32}


def _x(device, shape, dt):
    torch.manual_seed(58706)
    t = torch.randn(shape, dtype=torch.float32)
    return ttnn.from_torch(t, dtype=DT[dt], layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("dt", list(DT))
@pytest.mark.parametrize("shape", list(SHAPES))
def test_sum_h(device, shape, dt):
    ttnn.to_torch(ttnn.sum(_x(device, SHAPES[shape], dt), dim=-2, keepdim=True))


@pytest.mark.parametrize("dt", list(DT))
@pytest.mark.parametrize("shape", list(SHAPES))
def test_mean_h(device, shape, dt):
    ttnn.to_torch(ttnn.mean(_x(device, SHAPES[shape], dt), dim=-2, keepdim=True))


@pytest.mark.parametrize("dt", ["bf16"])
@pytest.mark.parametrize("shape", ["1x1x1024x1024", "4x1x2048x256", "1x1x256x2048"])
def test_moreh_sum_h(device, shape, dt):
    ttnn.to_torch(ttnn.operations.moreh.sum(_x(device, SHAPES[shape], dt), -2, keepdim=True))


@pytest.mark.parametrize("dt", ["bf16"])
@pytest.mark.parametrize("shape", ["1x1x1024x1024", "4x1x2048x256", "1x1x256x2048"])
def test_moreh_mean_h(device, shape, dt):
    ttnn.to_torch(ttnn.operations.moreh.mean(_x(device, SHAPES[shape], dt), dim=-2, keepdim=True))


@pytest.mark.parametrize("dt", ["bf16"])
@pytest.mark.parametrize("shape", ["1x1x1024x1024", "4x1x2048x256", "1x1x256x2048"])
def test_moreh_softmax_h(device, shape, dt):
    ttnn.to_torch(ttnn.operations.moreh.softmax(_x(device, SHAPES[shape], dt), -2))


FID = {"HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}
POOLS = {
    "320x48x48_k9": (1, 320, 48, 48, 9),
    "320x48x48_k36": (1, 320, 48, 48, 36),
    "64x96x96_k9": (1, 64, 96, 96, 9),
}


@pytest.mark.parametrize("fid", list(FID))
@pytest.mark.parametrize("case", list(POOLS))
def test_avg_pool_large(device, case, fid):
    # Large-window avg_pool2d: the reduce COL math over tilizeA_B unpacked tiles (must keep one face per bank)
    N, C, H, W, k = POOLS[case]
    torch.manual_seed(58706)
    t = torch.randn((N, C, H, W), dtype=torch.float32).permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False)
    out = ttnn.avg_pool2d(input_tensor=x, batch_size=N, input_h=H, input_w=W, channels=C, kernel_size=[k, k], stride=[2, 2],
                          padding=[1, 1], compute_kernel_config=cfg, config_tensor_in_dram=True)
    ttnn.to_torch(out)
