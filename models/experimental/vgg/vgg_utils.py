# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
from loguru import logger
from tt_lib.utils import pad_weight


def format_tensor(x, target_layout, device, output_mem_config, pad_value=0.0):
    if x.layout() == target_layout:
        return x
    if x.layout() == tt_lib.tensor.Layout.ROW_MAJOR and target_layout == tt_lib.tensor.Layout.TILE:
        x_padded_shape = tt_lib.tensor.pad_to_tile_shape(x.shape(), False, False, True, True)
        if x.shape() != x_padded_shape:
            return tt_lib.tensor.format_input_tensor(
                x, device, x_padded_shape, pad_value, target_layout, output_mem_config
            )
        else:
            return tt_lib.tensor.tilize(x, output_mem_config, use_multicore=True)
    elif x.layout() == tt_lib.tensor.Layout.TILE and target_layout == tt_lib.tensor.Layout.ROW_MAJOR:
        if x.shape() != x.shape_without_padding():
            return tt_lib.tensor.format_output_tensor(
                x, x.shape_without_padding(), device, target_layout, output_mem_config
            )
        else:
            return tt_lib.tensor.untilize(x, output_mem_config, use_multicore=True)
    else:
        assert False


def cache_weights_in_weka(device, model_location_generator, dtype=tt_lib.tensor.DataType.BFLOAT16):
    vgg_path = model_location_generator("vgg11", model_subdir="VGG")
    state_dict = torch.load(vgg_path / "vgg11.pth")

    file_name = "tt-metal/models/experimental/vgg/weights/"

    for key, value in state_dict.items():
        for i in range(4 - len(value.shape)):
            value = value.unsqueeze(0)
        if "classifier" in key:
            if "weight" in key:
                if "0" in key or "3" in key:
                    value = tt_lib.tensor.Tensor(
                        value.reshape(-1).tolist(),
                        value.shape,
                        dtype,
                        tt_lib.tensor.Layout.ROW_MAJOR,
                    ).to(tt_lib.tensor.Layout.TILE)
                else:
                    value = pad_weight(value)
                    value = tt_lib.tensor.Tensor(
                        value.reshape(-1).tolist(),
                        value.shape,
                        dtype,
                        tt_lib.tensor.Layout.ROW_MAJOR,
                    ).to(tt_lib.tensor.Layout.TILE)
            else:
                if "0" in key or "3" in key:
                    value = tt_lib.tensor.Tensor(
                        value.reshape(-1).tolist(),
                        value.shape,
                        dtype,
                        tt_lib.tensor.Layout.ROW_MAJOR,
                    )
                else:
                    extra_zeros = torch.zeros(1, 1, 1, 24)
                    value = torch.cat((value, extra_zeros), dim=-1)
                    value = tt_lib.tensor.Tensor(
                        value.reshape(-1).tolist(),
                        value.shape,
                        dtype,
                        tt_lib.tensor.Layout.ROW_MAJOR,
                    )
        else:
            value = tt_lib.tensor.Tensor(
                value.reshape(-1).tolist(),
                value.shape,
                dtype,
                tt_lib.tensor.Layout.ROW_MAJOR,
            )
        tt_lib.tensor.dump_tensor(file_name + str(key) + str(dtype) + ".bin", value)
