# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.gemma4.tt.vision.vision_ccl import tp_all_reduce
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
        # We TP across cluster axis 1 (all devices of the mesh).
        self.tp = args.tp

        if self.tp > 1:
            col_mapper = args.mesh_config.column_parallel(self.mesh_device)
            row_mapper = args.mesh_config.row_parallel(self.mesh_device)
        else:
            col_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
            row_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)

        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)
        pad_hidden_dim = lambda tensor, dim: pad_to_size(tensor, dim=dim, size=args.hidden_dim)
        torch_weight = lambda name: torch.transpose(
            self.state_dict[f"{state_dict_prefix}.{name}.linear.weight"], -2, -1
        )

        # The on-disk tensorbins hold the per-device shard, so the TP width is part
        # of the cache identity.
        tp_suffix = f"_tp{self.tp}" if self.tp > 1 else ""
        if args.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}{tp_suffix}"

        # Simplified tensor creation with DRAM memory config
        as_weight_tensor = lambda name, dims, mapper, type: ttnn.as_tensor(
            pad_hidden_dim(torch_weight(name[:]), dims[0]),
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name),
        )

        self.four_bit_mlp = args.optimizations.bfp4_mlp
        # ---- gate_proj and up_proj: column-sharded ----------------------------------------------------
        # Shape: [1, 1, dim, hidden_dim]; shard dim=-1 across cluster axis 1.
        mlp_dtype = ttnn.bfloat4_b if self.four_bit_mlp else ttnn.bfloat8_b
        self.gate_proj = as_weight_tensor("w1", (-1, -2), col_mapper, mlp_dtype)
        self.up_proj = as_weight_tensor("w3", (-1, -2), col_mapper, mlp_dtype)
        # ---- down_proj: row-sharded -------------------------------------------------------
        # Shape: [1, 1, hidden_dim, dim]; shard dim=-2 across cluster axis 1.
        # ``args.hidden_dim`` is padded to a multiple of tile_size * tp, and the padded
        # lanes are zero in both halves, so the per-device shards stay tile-aligned and
        # contribute nothing to the sum.
        self.down_proj = as_weight_tensor("w2", (-2, -1), row_mapper, ttnn.bfloat8_b)

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
        # full output dim.
        out = ttnn.linear(
            down_in,
            self.down_proj,
            compute_kernel_config=self.args.compute_kernel_config_lofi
            if self.four_bit_mlp
            else self.args.compute_kernel_config_hifi2_fp16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(down_in)

        original_shape = out.shape
        out = ttnn.reshape(
            out, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

        # Sum the partial sums so the block output is replicated again.
        return tp_all_reduce(out, self.args)
