# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch


def window_partition(input_feature, window_size, device):
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = ttnn.to_torch(input_feature)
    # 6D reshape and permute not supported in ttnn
    input_feature = torch.reshape(
        input_feature,
        (
            batch_size,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
            num_channels,
        ),
    )
    windows = torch.permute(input_feature, (0, 1, 3, 2, 4, 5))
    windows = torch.reshape(windows, (-1, window_size, window_size, num_channels))
    windows = ttnn.from_torch(windows, dtype=ttnn.bfloat16)
    return windows


def window_reverse(windows, window_size, height, width, device):
    num_channels = windows.shape[-1]

    windows = ttnn.to_torch(windows)
    # 6D reshape and permute not supported in ttnn
    windows = torch.reshape(
        windows,
        (
            -1,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            num_channels,
        ),
    )

    windows = torch.permute(windows, (0, 1, 3, 2, 4, 5))
    windows = torch.reshape(windows, (-1, height, width, num_channels))
    windows = ttnn.from_torch(windows, dtype=ttnn.bfloat16, device=device)

    return windows


def get_relative_position(config, device):
    window_size = 7
    coords_h = ttnn.arange(start=0, end=window_size, step=1)
    coords_w = ttnn.arange(start=0, end=window_size, step=1)
    coords_h = ttnn.reshape(coords_h, [window_size + 1])
    coords_w = ttnn.reshape(coords_w, [window_size + 1])
    coords_h = ttnn.to_torch(coords_h)
    coords_w = ttnn.to_torch(coords_w)
    coordinate = [coords_h[:window_size], coords_w[:window_size]]
    coords = torch.stack(torch.meshgrid(*coordinate, indexing="ij"))
    coords = ttnn.from_torch(coords, dtype=ttnn.bfloat16)
    coords_flatten = ttnn.reshape(coords, (coords.shape[0], -1))
    coords_flatten = ttnn.to_torch(coords_flatten)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = torch.permute(relative_coords, (1, 2, 0))
    relative_coords[:, :, 0:1] += window_size - 1
    relative_coords[:, :, 1:] += window_size - 1
    relative_coords[:, :, 0:1] *= 2 * window_size - 1
    relative_coords = ttnn.from_torch(relative_coords, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    relative_position_index = ttnn.sum(relative_coords, -1)
    relative_position_index = ttnn.to_torch(relative_position_index).long().squeeze(-1)
    bias_table = []
    for i_layer in range(len(config.depths)):
        relative_position_bias_table = torch.zeros(
            (2 * window_size - 1) * (2 * window_size - 1), config.num_heads[i_layer]
        )
        relative_position_bias = relative_position_bias_table[relative_position_index.reshape(-1)]
        relative_position_bias = torch.reshape(
            relative_position_bias,
            (window_size * window_size, window_size * window_size, -1),
        )
        relative_position_bias = torch.permute(relative_position_bias, (2, 0, 1))
        relative_position_bias = ttnn.from_torch(
            relative_position_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        bias_table.append(relative_position_bias)
    return bias_table


# def get_attn_mask()
def get_attn_mask(config, device):
    height = [56, 28, 14, 7]
    shift_size = [0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3]
    window_size = 7
    shift_idx = 0
    attention_mask_list = []
    for i in range(len(config.depths)):
        mask_list = []
        for j in range(config.depths[i]):
            if shift_size[shift_idx] > 0:
                # calculate attention mask for SW-MSA
                img_mask = torch.zeros((1, height[i], height[i], 1))
                height_slices = (
                    slice(0, -window_size),
                    slice(-window_size, -shift_size[shift_idx]),
                    slice(-shift_size[shift_idx], None),
                )
                width_slices = (
                    slice(0, -window_size),
                    slice(-window_size, -shift_size[shift_idx]),
                    slice(-shift_size[shift_idx], None),
                )

                count = 0
                for height_slice in height_slices:
                    for width_slice in width_slices:
                        img_mask[:, height_slice, width_slice, :] = count
                        count += 1

                img_mask = ttnn.from_torch(
                    img_mask, dtype=ttnn.bfloat16
                )  # , device = device, layout = ttnn.TILE_LAYOUT)
                mask_windows = window_partition(img_mask, window_size, device)

                mask_windows = ttnn.reshape(mask_windows, (-1, window_size * window_size, 1, 1))

                mask_windows = ttnn.to_torch(mask_windows)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

                attn_mask = ttnn.from_torch(attn_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                attn_mask = ttnn.to_device(attn_mask, device=device)
            else:
                attn_mask = None

            mask_list.append(attn_mask)
            shift_idx += 1

        attention_mask_list.append(mask_list)

    return attention_mask_list
