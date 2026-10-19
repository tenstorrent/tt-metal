# Scratch, not for merge: DRAM group_norm (groupnorm.cpp, the reduce helper's SCALAR statistics) at the shapes of the
# #58820 rounds and the SDXL VAE decoder's last block. Each test runs the op once; the output goes through ttnn.to_torch.
import pytest
import torch
import ttnn

CASES = {
    "vae_1x128x1024x1024_g32_ob32": (1, 128, 1024, 1024, 32, 32, 8, 8, ttnn.bfloat16),
    "1x2560x1x1024_g32_ob4": (1, 2560, 1, 1024, 32, 4, 8, 8, ttnn.bfloat16),
    "8x768x1x512_g32_ob2": (8, 768, 1, 512, 32, 2, 8, 8, ttnn.bfloat16),
    "1x1920x16x16_g32_ob1": (1, 1920, 16, 16, 32, 1, 4, 4, ttnn.bfloat16),
    "1x64x192x640_g16_ob10": (1, 64, 192, 640, 16, 10, 2, 4, ttnn.bfloat16),
    "1x64x64x64_g16_ob2": (1, 64, 64, 64, 16, 2, 2, 4, ttnn.bfloat16),
    "f32_1x2560x1x1024_g32_ob4": (1, 2560, 1, 1024, 32, 4, 8, 8, ttnn.float32),
    "f32_8x768x1x512_g32_ob2": (8, 768, 1, 512, 32, 2, 8, 8, ttnn.float32),
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("case", list(CASES))
def test_gn(device, case):
    N, C, H, W, groups, out_blocks, cy, cx, dtype = CASES[case]
    torch.manual_seed(61433)
    grid = ttnn.CoreGrid(y=cy, x=cx)
    t = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    wgt = torch.rand((C,), dtype=torch.bfloat16)
    bias = torch.rand((C,), dtype=torch.bfloat16)
    rm = ttnn.from_torch(t.permute(0, 2, 3, 1).reshape(N, 1, W * H, C), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT,
                         device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.tilize_with_zero_padding(rm, use_multicore=True)
    [g, b], mask = ttnn.dram_group_norm_params_from_torch([wgt, bias], C, groups, device, core_grid=grid, return_mask=True)
    out = ttnn.group_norm(x, num_groups=groups, input_mask=mask, weight=g, bias=b, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                          core_grid=grid, inplace=False, num_out_blocks=out_blocks, use_welford=False)
    ttnn.to_torch(out)
