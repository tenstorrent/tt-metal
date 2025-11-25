# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from .module import Module


class MoELayer(Module):

    """
    MoE layer
    """

    def __init__(self, mesh_device, state_dict, args, layer_num, dtype):
        super().__init__()
        self.mesh_device = mesh_device
        self.state_dict = state_dict
        self.args = args
        self.layer_num = layer_num
        self.dtype = dtype

    def forward(self, x: ttnn.Tensor, compute_kernel_config=None) -> ttnn.Tensor:
        return x
