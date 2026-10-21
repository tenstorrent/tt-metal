# Scratch, not for merge: the SCALAR SUM/AVG pair path at op level. DRAM group_norm (groupnorm.cpp, the reduce helper's SCALAR
# statistics) at bf16 HiFi2 and HiFi4, and the generic reduce's H and W reduction (sum and mean) over many tiles, which accumulates
# one block per core into one DEST tile. Each test runs the op once; the output goes through ttnn.to_torch.
import pytest
import torch
import ttnn

GN = {
    "vae_1x128x1024x1024_g32_ob32": (1, 128, 1024, 1024, 32, 32, 8, 8),
    "1x2560x1x1024_g32_ob4": (1, 2560, 1, 1024, 32, 4, 8, 8),
    "8x768x1x512_g32_ob2": (8, 768, 1, 512, 32, 2, 8, 8),
    "1x64x192x640_g16_ob10": (1, 64, 192, 640, 16, 10, 2, 4),
}
FID = {"HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("fid", list(FID))
@pytest.mark.parametrize("case", list(GN))
def test_gn(device, case, fid):
    N, C, H, W, groups, out_blocks, cy, cx = GN[case]
    torch.manual_seed(61433)
    grid = ttnn.CoreGrid(y=cy, x=cx)
    t = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    wgt = torch.rand((C,), dtype=torch.bfloat16)
    bias = torch.rand((C,), dtype=torch.bfloat16)
    rm = ttnn.from_torch(t.permute(0, 2, 3, 1).reshape(N, 1, W * H, C), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                         device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.tilize_with_zero_padding(rm, use_multicore=True)
    [g, b], mask = ttnn.dram_group_norm_params_from_torch([wgt, bias], C, groups, device, core_grid=grid, return_mask=True)
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False)
    out = ttnn.group_norm(x, num_groups=groups, input_mask=mask, weight=g, bias=b, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                          core_grid=grid, inplace=False, num_out_blocks=out_blocks, use_welford=False, compute_kernel_config=cfg)
    ttnn.to_torch(out)


HW = {
    "sum_bf16": (ttnn.sum, ttnn.bfloat16),
    "mean_bf16": (ttnn.mean, ttnn.bfloat16),
    "sum_f32": (ttnn.sum, ttnn.float32),
    "mean_f32": (ttnn.mean, ttnn.float32),
}
SHAPES = {"1x1x256x256": (1, 1, 256, 256), "1x1x1024x1024": (1, 1, 1024, 1024), "1x1x64x4096": (1, 1, 64, 4096), "1x8x96x96": (1, 8, 96, 96)}


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("case", list(HW))
def test_hw_big(device, case, shape):
    # The generic reduce's SCALAR path over many tiles
    op, dtype = HW[case]
    torch.manual_seed(61433)
    t = torch.randn(SHAPES[shape], dtype=torch.float32)
    x = ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.to_torch(op(x, dim=[2, 3], keepdim=True))
