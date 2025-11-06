# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
This is the modified version of the FeedForward for the Mistral-Small-3.1-24B-Instruct-2503 model.
This file implements the Vision FeedForward submodule specific for the Mistral-Small-3.1-24B-Instruct-2503 model.
"""

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule


class MistralTTVisionMLP(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        weight_cache_path,
        dtype,
        state_dict_prefix=None,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.args = args
        self.state_dict = state_dict
        self.dim = args.dim

        def get_weight(name):
            return torch.transpose(state_dict[f"{state_dict_prefix}{name}.weight"], -2, -1)

        def get_bias(name):
            return state_dict[f"{state_dict_prefix}{name}.bias"]

        def cache_name(name):
            if args.dummy_weights:
                return None
            return weight_cache_path / f"{state_dict_prefix}.{name}"

        def as_tensor(name, dtype, is_bias=False):
            tensor_data = get_bias(name) if is_bias else get_weight(name)
            return ttnn.as_tensor(
                tensor_data,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Weights and Biases
        self.w1 = as_tensor("w1", dtype)
        self.b1 = as_tensor("w1", ttnn.bfloat16, is_bias=False)

        self.w3 = as_tensor("w3", dtype)
        self.b3 = as_tensor("w3", ttnn.bfloat16, is_bias=False)

        self.w2 = as_tensor("w2", dtype)
        self.b2 = as_tensor("w2", ttnn.bfloat16, is_bias=False)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            dst_full_sync_en=False,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Qwen HF MLP reference:
        output = down_proj(act_fn(gate_proj(x)) * up_proj(x))
        Mapping:
            w1 -> gate_proj
            w3 -> up_proj
            w2 -> down_proj
        """

        # Linear with SILU activation
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            activation="silu",
            compute_kernel_config=self.compute_kernel_config,
        )

        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Element-wise multiply
        w2_in = ttnn.mul(w1_out, w3_out, dtype=ttnn.bfloat16)

        # Final projection
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        ttnn.deallocate(w2_in)
        return w2_out
