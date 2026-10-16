# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B RMSNorm.

Standard (non-distributed) RMSNorm: ``out = x_normed * weight``. For Llama with TP=4, hidden=4096,
local_hidden=1024. The hidden state is replicated on the TP axis after the allreduce, so we use the
non-distributed norm with a replicated weight."""

from torch import nn

import ttnn
from models.demos.llama_3_1_8b.tt.config import MeshConfig
from models.demos.llama_3_1_8b.utils.general_utils import get_cache_file_name


class RMSNorm(nn.Module):
    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None, mesh_config=None):
        super().__init__()
        if state_dict:
            weight = state_dict["weight"]
            torch_weight = weight.reshape((1, 1, -1, ttnn.TILE_SIZE))
        else:
            torch_weight = None

        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
        # Non-distributed: weight is replicated across the mesh (not sharded). The hidden state
        # is replicated on the TP axis after the allreduce, so we use plain rms_norm.
        self.tt_weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=None,
        )

        self.eps = hf_config.rms_norm_eps
        self.mesh_device = mesh_device

    def forward(self, x):
        tt_output = ttnn.rms_norm(
            x,
            weight=self.tt_weight,
            epsilon=self.eps,
        )
        return tt_output
