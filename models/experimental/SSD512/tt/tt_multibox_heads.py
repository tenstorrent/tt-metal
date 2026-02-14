# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from models.experimental.SSD512.tt.utils import Conv2dOperation


# Multi-box detection head for location or confidence prediction at each feature scale
class TtMultiBoxHEAD:
    def __init__(self, conv_config_layer, device, activation_layer=None):
        self.device = device

        self.layer = Conv2dOperation(
            device=device,
            conv_config=conv_config_layer,
            activation_layer=activation_layer,
        )

    # Forward pass through detection head layer
    def __call__(self, device, input):
        result = self.layer(device, input)

        return result
