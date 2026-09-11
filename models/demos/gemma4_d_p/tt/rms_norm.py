# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from torch import nn

import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.utils.general_utils import get_cache_file_name


class RMSNorm(nn.Module):
    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None, mesh_config=None, with_scale=True):
        super().__init__()
        self.with_scale = with_scale

        if with_scale and state_dict and "weight" in state_dict:
            torch_weight = state_dict["weight"].reshape((1, 1, -1, ttnn.TILE_SIZE))
        else:
            torch_weight = None

        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape)

        if with_scale:
            self.tt_weight = ttnn.as_tensor(
                torch_weight,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # TILE gamma enables the large-tensor RMSNorm path, which bounds L1 usage
            # for the FP32 intermediate buffers in the 5376-wide prefill norm.
            flat_weight = ttnn.reshape(self.tt_weight, (1, 1, 1, hf_config.hidden_size))
            self.tt_weight = ttnn.to_layout(flat_weight, ttnn.TILE_LAYOUT)
        else:
            self.tt_weight = None

        self.eps = hf_config.rms_norm_eps
        self.mesh_device = mesh_device
        # Match the reference's FP32 prefill RMSNorm computation.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def forward(self, x):
        return ttnn.rms_norm(
            x,
            weight=self.tt_weight,
            epsilon=self.eps,
            compute_kernel_config=self.compute_kernel_config,
        )
