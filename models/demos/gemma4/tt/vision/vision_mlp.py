# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Tensor-parallel ("Megatron-style") Gemma-4 vision MLP.

Mirrors the LLM TP convention from `tt_transformers.tt.mlp`:

  in:  replicated x (the wrapping DistributedNorm produced this)
  gate/up: column-sharded W1, W3  ──▶  GELU * up     (no comm)
  down:    row-sharded   W2       ──▶  partial sums
                                  ──▶ reduce_scatter along TP
  out: fractured along dim=3 (each device owns dim/TP); on 2D also DP-sharded on batch
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.gemma4.tt.vision.vision_model_config import vision_tp_reduce_scatter
from models.tt_transformers.tt.common import Mode, pad_to_size


class Gemma4VisionMLP(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        state_dict_prefix=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.dim = args.dim
        self.cluster_shape = args.cluster_shape
        # We TP across cluster axis 1. DP (when present) is axis 0.
        self.tp = args.tp

        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)
        pad_hidden_dim = lambda tensor, dim: pad_to_size(tensor, dim=dim, size=args.hidden_dim)
        torch_weight = lambda name: torch.transpose(
            self.state_dict[f"{state_dict_prefix}.{name}.linear.weight"], -2, -1
        )

        if args.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}.tp{self.tp}"

        def as_weight_tensor(name, pad_dim, shard_dim, dtype):
            weight = pad_hidden_dim(torch_weight(name), pad_dim).unsqueeze(0).unsqueeze(0)
            return ttnn.as_tensor(
                weight,
                dtype=dtype,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(None, shard_dim), mesh_shape=self.cluster_shape
                ),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name(name),
            )

        self.four_bit_mlp = args.optimizations.bfp4_mlp
        col_dtype = ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b
        # gate/up: [1, 1, dim, hidden_dim]; shard dim=-1 across cluster axis 1.
        self.gate_proj = as_weight_tensor("w1", pad_dim=-1, shard_dim=-1, dtype=col_dtype)
        self.up_proj = as_weight_tensor("w3", pad_dim=-1, shard_dim=-1, dtype=col_dtype)
        # down: [1, 1, hidden_dim, dim]; shard dim=-2 across cluster axis 1.
        self.down_proj = as_weight_tensor("w2", pad_dim=-2, shard_dim=-2, dtype=ttnn.bfloat8_b)

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        seq_len = x.shape[-2]
        if seq_len >= 512:
            x = ttnn.reshape(x, [1, seq_len // 512, 512, -1])

        # Column-sharded gate/up: output is fractured along the intermediate dim.
        gate_out = ttnn.linear(
            x,
            self.gate_proj,
            activation="gelu_approx",
            compute_kernel_config=self.args.compute_kernel_config_lofi
            if self.four_bit_mlp
            else self.args.compute_kernel_config_hifi2_fp16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        up_out = ttnn.linear(
            x,
            self.up_proj,
            compute_kernel_config=self.args.compute_kernel_config_lofi
            if self.four_bit_mlp
            else self.args.compute_kernel_config_hifi2_fp16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        down_in = ttnn.mul(up_out, gate_out)
        ttnn.deallocate(up_out)
        ttnn.deallocate(gate_out)

        # Row-sharded down: each device computes a partial sum of the full output dim.
        out_partial = ttnn.linear(
            down_in,
            self.down_proj,
            compute_kernel_config=self.args.compute_kernel_config_lofi
            if self.four_bit_mlp
            else self.args.compute_kernel_config_hifi2_fp16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(down_in)

        # Reduce-scatter along TP: fractured along dim=3 -- the vision-block I/O contract.
        out = vision_tp_reduce_scatter(
            out_partial,
            self.mesh_device,
            self.tt_ccl,
            self.args,
        )
        if out is not out_partial:
            ttnn.deallocate(out_partial)

        original_shape = out.shape
        return ttnn.reshape(
            out, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
