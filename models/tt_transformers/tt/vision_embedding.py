# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.common.lightweightmodule import LightweightModule


class VisionEmbedding(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        dtype,
        image_size,
        patch_size,
        num_channels,
        hidden_dim,
        bias=True,
    ):
        super().__init__()

    def forward(self, x):
        return 0
