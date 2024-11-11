# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch


def window_partition(input_feature, window_size, mesh_mapper, device, output_mesh_composer):
    input_feature = ttnn.to_torch(input_feature, mesh_composer=output_mesh_composer)
    batch_size, height, width, num_channels = input_feature.shape
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
    windows = ttnn.from_torch(windows, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)  # , device = device)
    return windows


def window_reverse(windows, window_size, height, width, device, mesh_mapper, output_mesh_composer):
    num_channels = windows.shape[-1]

    windows = ttnn.to_torch(windows, mesh_composer=output_mesh_composer)
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
    windows = ttnn.from_torch(windows, dtype=ttnn.bfloat16, device=device, mesh_mapper=mesh_mapper)

    return windows


def get_relative_position(config, parameters, mesh_mapper, device, output_mesh_composer):
    window_size = 7

    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    cords = [coords_h, coords_w]
    coords = torch.stack(torch.meshgrid(*cords, indexing="ij"))

    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()

    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1
    relative_position_index = relative_coords.sum(-1)

    relative_pos_table = {}
    depth = 2
    for i in range(4):
        if i == 2:
            depth = 6
        else:
            depth = 2
        j = 0
        bias_table = []
        while j < depth:
            relative_position_bias_table = ttnn.to_torch(
                parameters.encoder.layers[i].blocks[j].attention.relative_position_bias_table,
                mesh_composer=output_mesh_composer,
            )

            relative_position_bias = relative_position_bias_table[relative_position_index.view(-1)]
            relative_position_bias = relative_position_bias.view(
                window_size * window_size, window_size * window_size, -1
            )

            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            relative_position_bias = ttnn.from_torch(
                relative_position_bias.unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=mesh_mapper,
            )

            bias_table.append(relative_position_bias)
            j += 1
        relative_pos_table[i] = bias_table
    return relative_pos_table


def get_attn_mask(config, mesh_mapper, device, output_mesh_composer, batch_size):
    height = [56, 28, 14, 7]
    shift_size = [0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3]
    window_size = 7
    shift_idx = 0
    attention_mask_list = []
    for i in range(len(config.depths)):
        mask_list = []
        for j in range(config.depths[i]):
            if shift_size[shift_idx] > 0 and i != 3:
                # calculate attention mask for SW-MSA
                img_mask = torch.zeros((batch_size, height[i], height[i], 1))
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

                img_mask = ttnn.from_torch(img_mask, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
                mask_windows = window_partition(img_mask, window_size, mesh_mapper, device, output_mesh_composer)

                mask_windows = ttnn.reshape(mask_windows, (-1, window_size * window_size, 1, 1))

                mask_windows = ttnn.to_torch(mask_windows, mesh_composer=output_mesh_composer)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

                attn_mask = torch.reshape(attn_mask, (attn_mask.shape[0], 1, attn_mask.shape[1], attn_mask.shape[2]))
                attn_mask = ttnn.from_torch(
                    attn_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper, device=device
                )

            else:
                attn_mask = None

            mask_list.append(attn_mask)
            shift_idx += 1

        attention_mask_list.append(mask_list)

    return attention_mask_list
