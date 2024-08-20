# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
import ttnn
from tt_lib.utils import pad_weight

from pathlib import Path
from torchvision import models


def format_tensor(x, target_layout, device, output_mem_config, pad_value=0.0):
    if x.get_layout() == target_layout:
        return x

    if x.get_layout() == ttnn.ROW_MAJOR_LAYOUT and target_layout == ttnn.TILE_LAYOUT:
        x_padded_shape = ttnn.pad_to_tile_shape(x.get_legacy_shape(), False, False, True, True)
        if x.get_legacy_shape() != x_padded_shape:
            return ttnn.format_input_tensor(x, device, x_padded_shape, pad_value, target_layout, output_mem_config)
        else:
            return ttnn.tilize(x, memory_config=output_mem_config, use_multicore=True)
    elif x.get_layout() == ttnn.TILE_LAYOUT and target_layout == ttnn.ROW_MAJOR_LAYOUT:
        if x.get_legacy_shape() != x.shape_without_padding():
            return ttnn.format_output_tensor(x, x.shape_without_padding(), device, target_layout, output_mem_config)
        else:
            return ttnn.untilize(x, memory_config=output_mem_config, use_multicore=True)
    else:
        assert False


def cache_weights_in_weka(device, model_location_generator):
    vgg_path = model_location_generator("vgg16", model_subdir="VGG")
    state_dict = torch.load(vgg_path / "vgg16.pth")

    file_name = "tt-metal/models/experimental/vgg/weights/"

    for key, value in state_dict.items():
        for i in range(4 - len(value.shape)):
            value = value.unsqueeze(0)
        if "classifier" in key:
            if "weight" in key:
                if "0" in key or "3" in key:
                    value = ttnn.Tensor(
                        value.reshape(-1).tolist(),
                        value.shape,
                        ttnn.bfloat16,
                        ttnn.ROW_MAJOR_LAYOUT,
                    ).to(ttnn.TILE_LAYOUT)
                else:
                    value = pad_weight(value)
                    value = ttnn.Tensor(
                        value.reshape(-1).tolist(),
                        value.shape,
                        ttnn.bfloat16,
                        ttnn.ROW_MAJOR_LAYOUT,
                    ).to(ttnn.TILE_LAYOUT)
            else:
                if "0" in key or "3" in key:
                    value = ttnn.Tensor(
                        value.reshape(-1).tolist(),
                        value.shape,
                        ttnn.bfloat16,
                        ttnn.ROW_MAJOR_LAYOUT,
                    )
                else:
                    extra_zeros = torch.zeros(1, 1, 1, 24)
                    value = torch.cat((value, extra_zeros), dim=-1)
                    value = ttnn.Tensor(
                        value.reshape(-1).tolist(),
                        value.shape,
                        ttnn.bfloat16,
                        ttnn.ROW_MAJOR_LAYOUT,
                    )
        else:
            value = ttnn.Tensor(
                value.reshape(-1).tolist(),
                value.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
        ttnn.dump_tensor(file_name + str(key) + ".bin", value)


def store_weights(model_version, file_name, dtype, base_addresses):
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    state_dict = model.state_dict()

    for key, value in state_dict.items():
        for base_address in base_addresses:
            if base_address == "" or key.startswith(base_address):
                if key.endswith((".weight", ".bias")) and not os.path.exists(
                    file_name + str(key) + str(dtype) + ".bin"
                ):
                    torch.save(value, f"{file_name}{key}{dtype}.bin")
                    continue

        while len(value.shape) < 4:
            value = value.unsqueeze(0)

        if value.shape[-2] % 32 == 0 and value.shape[-1] % 32 == 0:
            value = ttnn.Tensor(
                value.reshape(-1).tolist(),
                value.shape,
                dtype,
                ttnn.ROW_MAJOR_LAYOUT,
            ).to(ttnn.TILE_LAYOUT)
        else:
            value = ttnn.Tensor(
                value.reshape(-1).tolist(),
                value.shape,
                dtype,
                ttnn.ROW_MAJOR_LAYOUT,
            )

        ttnn.dump_tensor(file_name + str(key) + str(dtype) + ".bin", value)


def get_tt_cache_path(model_version):
    tt_cache_path = Path("/mnt/MLPerf/tt_dnn-models/tt/VGG") / model_version

    if tt_cache_path.exists():
        return str(tt_cache_path) + "/"
    else:
        Path(f"models/experimental/vgg/datasets/{model_version}").mkdir(parents=True, exist_ok=True)
        return str(Path(f"models/experimental/vgg/datasets/{model_version}")) + "/"
