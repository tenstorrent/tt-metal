# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def test_group_norm_large_ex_external_cb(device):
    torch.manual_seed(0)
    shape = (1, 1, 1280 * 720, 256)  # [N, 1, H*W, C]
    num_groups = 32
    eps = 1e-5

    input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    weight = torch.randn((shape[-1],), dtype=torch.bfloat16)
    bias = torch.randn((shape[-1],), dtype=torch.bfloat16)
    c = shape[-1]
    weight_4d = weight.reshape(1, 1, c // 32, 32)
    bias_4d = bias.reshape(1, 1, c // 32, 32)

    # GroupNorm golden: convert [N,1,H*W,C] -> [N,C,1,H*W], apply GN, convert back.
    input_tensor_nchw = input_tensor.permute(0, 3, 1, 2).float()
    golden = torch.nn.functional.group_norm(
        input_tensor_nchw, num_groups=num_groups, weight=weight.float(), bias=bias.float(), eps=eps
    ).permute(0, 2, 3, 1)

    input_tensor_tt = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    w_tt = ttnn.from_torch(weight_4d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_tt = ttnn.from_torch(bias_4d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    sharded_mem_config, grid_size = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
        device=device,
        num_channels=c,
        num_groups=num_groups,
        input_nhw=1280 * 720,
        is_height_sharded=False,
        is_row_major=False,
    )
    output_tensor_tt = ttnn.group_norm(
        input_tensor_tt,
        num_groups=num_groups,
        epsilon=eps,
        weight=w_tt,
        bias=b_tt,
        core_grid=grid_size,
        inplace=False,
        num_out_blocks=-1,
    )
    output_tensor = ttnn.to_torch(output_tensor_tt)
    assert_with_pcc(golden, output_tensor)
