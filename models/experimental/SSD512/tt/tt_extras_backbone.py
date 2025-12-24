# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.SSD512.tt.utils import Conv2dOperation


class TtExtrasBackbone:
    def __init__(self, conv_config_layer, device, batch_size: int):
        self.batch_size = batch_size
        self.device = device

        layers = []
        for conv_config in conv_config_layer:
            layers.append(
                Conv2dOperation(
                    device=device,
                    conv_config=conv_config,
                    activation_layer=ttnn.relu,
                )
            )

        self.block = layers

    def __call__(self, device, input, return_residual_sources=False):
        residual_sources = []
        for i, layer in enumerate(self.block):
            if i == 0:
                result = layer(device, input)
            else:
                result = layer(device, result)

            if i % 2 == 1:
                residual_sources.append(result)

        if return_residual_sources:
            return result, residual_sources
        return result
