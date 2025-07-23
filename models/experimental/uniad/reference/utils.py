# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn.functional as F


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    im2col_step: torch.Tensor,
) -> torch.Tensor:
    bs, num_keys, num_heads, head_dim = value.shape
    num_levels = value_spatial_shapes.shape[0]
    num_queries = sampling_locations.shape[1]
    num_points = sampling_locations.shape[4]

    # Split value into a list of tensors for each level
    value_list = []
    start = 0
    for lvl in range(num_levels):
        h_l, w_l = value_spatial_shapes[lvl]
        h_l = int(h_l.item())
        w_l = int(w_l.item())
        len_l = h_l * w_l
        value_l = value[:, start : start + len_l, :, :]
        value_list.append(value_l)
        start += len_l

    # Normalize sampling locations to [-1, 1]
    sampling_grids = []
    for lvl in range(num_levels):
        h_l, w_l = value_spatial_shapes[lvl]
        h_l = int(h_l.item())
        w_l = int(w_l.item())
        grid = sampling_locations[:, :, :, lvl, :, :]
        grid = grid.clone()
        grid[..., 0] = grid[..., 0] / w_l * 2 - 1
        grid[..., 1] = grid[..., 1] / h_l * 2 - 1
        sampling_grids.append(grid)

    # Perform sampling and attention
    output = torch.zeros(bs, num_queries, num_heads, head_dim, device=value.device)
    for lvl in range(num_levels):
        h_l, w_l = value_spatial_shapes[lvl]
        h_l = int(h_l.item())
        w_l = int(w_l.item())
        value_l = value_list[lvl].permute(0, 2, 3, 1).reshape(bs * num_heads, head_dim, h_l, w_l)
        grid = sampling_grids[lvl].permute(0, 2, 1, 3, 4).reshape(bs * num_heads, num_queries * num_points, 1, 2)
        sampled = F.grid_sample(value_l, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.view(bs, num_heads, head_dim, num_queries, num_points).permute(0, 3, 1, 4, 2)
        attn = attention_weights[:, :, :, lvl, :].unsqueeze(-1)
        output += (sampled * attn).sum(-2)

    return output.view(bs, num_queries, num_heads * head_dim)
