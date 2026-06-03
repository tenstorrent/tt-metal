# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
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
        # We TP across cluster axis 1 (the row axis on T3K, i.e. all 8 devices).
        self.tp = self.cluster_shape[1]
        # For T3K (1, 8), `tt_all_reduce` ignores `cluster_axis` and falls
        # straight into `reduce_scatter_minimal_async` over the non-1 axis,
        # so any cluster_axis other than 1 (which is short-circuited) works.
        self.ccl_cluster_axis = 0

        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)
        pad_hidden_dim = lambda tensor, dim: pad_to_size(tensor, dim=dim, size=args.hidden_dim)
        torch_weight = lambda name: torch.transpose(
            self.state_dict[f"{state_dict_prefix}.{name}.linear.weight"], -2, -1
        )

        if args.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        # Simplified tensor creation with DRAM memory config
        as_weight_tensor = lambda name, dims, type: ttnn.as_tensor(
            pad_hidden_dim(torch_weight(name[:]), dims[0]),
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name),
        )

        self.four_bit_mlp = args.optimizations.bfp4_mlp
        # ---- gate_proj and up_proj: column-sharded ----------------------------------------------------
        # Shape: [1, 1, dim, hidden_dim]; shard dim=-1 across cluster axis 1.
        print(self.state_dict)
        self.gate_proj = as_weight_tensor("w1", (-1, -2), ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b)
        self.up_proj = as_weight_tensor("w3", (-1, -2), ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b)
        # ---- down_proj: row-sharded -------------------------------------------------------
        # Shape: [1, 1, hidden_dim, dim]; shard dim=-2 across cluster axis 1.
        self.down_proj = as_weight_tensor("w2", (-2, -1), ttnn.bfloat8_b)

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        """
        HF reference: self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))
        """
        seq_len = x.shape[-2]
        if seq_len >= 512:
            x = ttnn.reshape(x, [1, seq_len // 512, 512, -1])

        # fc1: column-sharded matmul + bias + GELU. Output is column-sharded
        # along the intermediate dim; no comm yet.
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

        # fc2: row-sharded matmul. Each device computes a partial sum of the
        # full output dim. We fold the bias in *after* the all-reduce.
        out = ttnn.linear(
            down_in,
            self.down_proj,
            compute_kernel_config=self.args.compute_kernel_config_lofi
            if self.four_bit_mlp
            else self.args.compute_kernel_config_hifi2_fp16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(down_in)

        # # On T3K (1, 8) `tt_all_reduce(dim=3)` is implemented as a
        # # reduce_scatter, so the result is fractured along dim=3 -- exactly
        # # the block I/O contract that the LLM uses.
        # ttnn.synchronize_device(self.mesh_device)
        # out = tt_all_reduce(
        #     out_partial,
        #     self.mesh_device,
        #     self.tt_ccl,
        #     cluster_axis=self.ccl_cluster_axis,
        #     dim=3,
        #     sharded=False,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     dtype=self.args.ccl_dtype,
        #     topology=self.args.ccl_topology(),
        # )
        # if out is not out_partial:
        #     ttnn.deallocate(out_partial)

        original_shape = out.shape
        return ttnn.reshape(
            out, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
