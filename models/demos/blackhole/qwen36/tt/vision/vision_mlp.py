# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tensor-parallel ("Megatron-style") qwen35_27b vision MLP.

Mirrors the LLM TP convention from `tt_transformers.tt.mlp`:

  in:  replicated x (the wrapping DistributedLayerNorm produced this)
  fc1: column-sharded W1, b1  ──▶  GELU            (no comm)
  fc2: row-sharded   W2       ──▶  partial sums
                              ──▶  tt_all_reduce(dim=3)  (reduce_scatter on T3K/QB2)
                              ──▶  + b2 (sharded along dim=3)
  out: fractured along dim=3 (each device owns dim/TP)

The fractured output then re-enters the next block's DistributedLayerNorm,
which gathers it back to replicated.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode, pad_to_size


class MLP(LightweightModule):
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
        # We TP across cluster axis 1 (the row axis on T3K/QB2).
        self.tp = self.cluster_shape[1]
        # For T3K/QB2, `tt_all_reduce` ignores `cluster_axis` and falls
        # straight into `reduce_scatter_minimal_async` over the non-1 axis,
        # so any cluster_axis other than 1 (which is short-circuited) works.
        self.ccl_cluster_axis = 0

        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)

        pad_hidden_dim = lambda tensor, dim: pad_to_size(tensor, dim=dim, size=args.hidden_dim)
        torch_weight = lambda name: torch.transpose(self.state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
        torch_bias = lambda name: self.state_dict[f"{state_dict_prefix}.{name}.bias"]

        if args.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}.tp{self.tp}"

        # ---- fc1: column-sharded ----------------------------------------------------
        # torch_weight("linear_fc1") has shape [dim, intermediate]. We pad the
        # output dim up to args.hidden_dim, then shard along the output dim.
        fc1_w = pad_hidden_dim(torch_weight("linear_fc1"), dim=-1).unsqueeze(0).unsqueeze(0)
        # Shape: [1, 1, dim, hidden_dim]; shard dim=-1 across cluster axis 1.
        self.linear_fc1_weight = ttnn.as_tensor(
            fc1_w,
            dtype=ttnn.bfloat4_b if args.optimizations.bfp4_mlp else ttnn.bfloat8_b,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -1), mesh_shape=self.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("linear_fc1_w_col"),
        )

        fc1_b = pad_hidden_dim(torch_bias("linear_fc1"), dim=-1)
        # 1-D bias [hidden_dim]; shard along its only dim across axis 1.
        self.linear_fc1_bias = ttnn.as_tensor(
            fc1_b,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -1), mesh_shape=self.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("linear_fc1_b_col"),
        )

        # ---- fc2: row-sharded -------------------------------------------------------
        # torch_weight("linear_fc2") has shape [intermediate, dim]. Pad input dim
        # up to args.hidden_dim and shard along the input dim.
        fc2_w = pad_hidden_dim(torch_weight("linear_fc2"), dim=-2).unsqueeze(0).unsqueeze(0)
        # Shape: [1, 1, hidden_dim, dim]; shard dim=-2 across cluster axis 1.
        self.linear_fc2_weight = ttnn.as_tensor(
            fc2_w,
            dtype=ttnn.bfloat8_b,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -2), mesh_shape=self.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("linear_fc2_w_row"),
        )

        # The MLP output is fractured along dim=3 (post reduce_scatter), so
        # the bias is sharded along the same axis.
        self.linear_fc2_bias = ttnn.as_tensor(
            torch_bias("linear_fc2"),
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -1), mesh_shape=self.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("linear_fc2_b_frac"),
        )

        self.four_bit_mlp = args.optimizations.bfp4_mlp

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        """
        HF reference: self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))
        """
        seq_len = x.shape[-2]
        if seq_len >= 1024:
            x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])

        # fc1: column-sharded matmul + bias + GELU. Output is column-sharded
        # along the intermediate dim; no comm yet.
        w1_out = ttnn.linear(
            x,
            self.linear_fc1_weight,
            bias=self.linear_fc1_bias,
            activation="gelu",
            compute_kernel_config=self.args.compute_kernel_config_lofi
            if self.four_bit_mlp
            else self.args.compute_kernel_config_hifi2_fp16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # fc2: row-sharded matmul. Each device computes a partial sum of the
        # full output dim. We fold the bias in *after* the all-reduce.
        w2_partial = ttnn.linear(
            w1_out,
            self.linear_fc2_weight,
            compute_kernel_config=self.args.compute_kernel_config_lofi
            if self.four_bit_mlp
            else self.args.compute_kernel_config_hifi2_fp16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w1_out)

        # On T3K/QB2 `tt_all_reduce(dim=3)` is implemented as a
        # reduce_scatter, so the result is fractured along dim=3 -- exactly
        # the block I/O contract that the LLM uses.
        w2_frac = tt_all_reduce(
            w2_partial,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=self.ccl_cluster_axis,
            dim=3,
            sharded=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.args.ccl_dtype,
            topology=self.args.ccl_topology(),
        )
        if w2_frac is not w2_partial:
            ttnn.deallocate(w2_partial)

        # Bias is also fractured along dim=3 to match.
        out = ttnn.add(w2_frac, self.linear_fc2_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if out is not w2_frac:
            ttnn.deallocate(w2_frac)

        original_shape = out.shape
        return ttnn.reshape(
            out, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
