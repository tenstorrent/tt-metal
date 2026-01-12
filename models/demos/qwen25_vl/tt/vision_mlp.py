# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import pad_to_size


class MLP(LightweightModule):
    def __init__(self, mesh_device, args, state_dict, weight_cache_path, layer_num, state_dict_prefix=None):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.dim = args.vision_dim
        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)
        pad_hidden_dim = lambda tensor, dim: pad_to_size(tensor, dim=dim, size=args.vision_hidden_dim)
        torch_weight = lambda name: torch.transpose(self.state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
        torch_bias = lambda name: self.state_dict[f"{state_dict_prefix}.{name}.bias"]

        if args.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        # Simplified tensor creation with DRAM memory config
        as_weight_tensor = lambda name, dims, type: ttnn.as_tensor(
            pad_hidden_dim(torch_weight(name[:2]), dims[0]),
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name),
        )

        # Create bias tensors
        as_bias_tensor = lambda name, pad: ttnn.as_tensor(
            pad_hidden_dim(torch_bias(name[:2]), dim=-1) if pad else torch_bias(name[:2]),
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name),
        )

        self.four_bit_mlp = args.optimizations.bfp4_mlp

        # Create weights with appropriate precision
        self.w1 = as_weight_tensor("w1", (-1, -2), ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b)
        # self.w1.shape = [640, 3456] each device of N300 sharded on dim=0
        self.w2 = as_weight_tensor("w2", (-2, -1), ttnn.bfloat8_b)
        # self.w2.shape = [3456, 640] each device of N300 sharded on dim=1
        self.w3 = as_weight_tensor("w3", (-1, -2), ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b)
        # self.w3.shape = [640, 3456] each device of N300 sharded on dim=0

        # Create bias tensors
        self.w1_bias = as_bias_tensor("w1_bias", pad=True)
        self.w2_bias = as_bias_tensor("w2_bias", pad=False)
        self.w3_bias = as_bias_tensor("w3_bias", pad=True)

    def forward(self, x: ttnn.Tensor, mode) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        if seq_len >= 1024:
            x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])

        # Linear projections with bias
        w1_out = ttnn.linear(
            x,
            self.w1,
            bias=self.w1_bias,
            compute_kernel_config=self.args.compute_kernel_config_lofi
            if self.four_bit_mlp
            else self.args.compute_kernel_config_hifi2_fp16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        w3_out = ttnn.linear(
            x,
            self.w3,
            bias=self.w3_bias,
            compute_kernel_config=self.args.compute_kernel_config_lofi
            if self.four_bit_mlp
            else self.args.compute_kernel_config_hifi2_fp16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        # Multiply with SiLU activation
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        # Final projection with bias
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            bias=self.w2_bias,
            compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w2_in)

        original_shape = w2_out.shape
        return ttnn.reshape(
            w2_out, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
