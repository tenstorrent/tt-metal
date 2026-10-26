# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_gather, tt_all_reduce
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

        # Tensor parallel (args.vision_tp): w1/w3 split on the hidden dim, w2 on its input dim, the w2
        # partial sums reduced across chips; otherwise every chip holds the whole MLP.
        self.tensor_parallel = args.vision_tp_devices > 1
        tp_name = lambda name: f"{name}_tp{args.vision_tp_devices}" if self.tensor_parallel else name
        shard_or_replicate = lambda dim: (
            ttnn.ShardTensorToMesh(self.mesh_device, dim=dim)
            if self.tensor_parallel
            else ttnn.ReplicateTensorToMesh(self.mesh_device)
        )

        # Simplified tensor creation with DRAM memory config; dims[0] is the padded (hidden) dim, which is
        # the dim split over the chips
        as_weight_tensor = lambda name, dims, type: ttnn.as_tensor(
            pad_hidden_dim(torch_weight(name[:2]), dims[0]),
            dtype=type,
            device=self.mesh_device,
            mesh_mapper=shard_or_replicate(dims[0]),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(tp_name(name)),
        )

        # Create bias tensors (the padded w1/w3 biases follow their split; the w2 bias is added once after the reduce)
        as_bias_tensor = lambda name, pad: ttnn.as_tensor(
            pad_hidden_dim(torch_bias(name[:2]), dim=-1) if pad else torch_bias(name[:2]),
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=shard_or_replicate(-1) if pad else ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(tp_name(name) if pad else name),
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

    # Rows per MLP pass. The MLP is row-wise, so a long patch sequence is run in slices: the two bf16
    # projections of a full 3584x3584 image (67k rows x 3456) would otherwise hold ~0.9 GB of DRAM per
    # chip at once, which with a large KV pool (72B, 128k context) does not fit. Numerics are unchanged.
    MAX_ROWS_PER_PASS = 8192

    def forward(self, x: ttnn.Tensor, mode) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        if seq_len > self.MAX_ROWS_PER_PASS and seq_len % 1024 == 0:
            hidden = x.shape[-1]
            outs = []
            for start in range(0, seq_len, self.MAX_ROWS_PER_PASS):
                end = min(start + self.MAX_ROWS_PER_PASS, seq_len)
                x_chunk = ttnn.slice(x, [0, 0, start, 0], [1, 1, end, hidden])
                outs.append(self._forward_rows(x_chunk, mode))  # consumes x_chunk
            ttnn.deallocate(x)
            out = ttnn.concat(outs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for o in outs:
                ttnn.deallocate(o)
            return out
        return self._forward_rows(x, mode)

    def _forward_rows(self, x: ttnn.Tensor, mode) -> ttnn.Tensor:
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
            bias=None if self.tensor_parallel else self.w2_bias,
            compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w2_in)
        if self.tensor_parallel:
            # per-chip partial sums over this chip's hidden slice -> full projection on every chip, then the bias once
            reduced = tt_all_reduce(
                w2_out,
                self.mesh_device,
                self.args.get_tt_ccl(),
                cluster_axis=0,
                dim=3,
                topology=self.args.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            w2_out = tt_all_gather(
                reduced,
                self.mesh_device,
                self.args.get_tt_ccl(),
                cluster_axis=None,  # 1x8 mesh: gather over the whole mesh
                dim=3,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.args.ccl_topology(),
                dtype=ttnn.bfloat16,
            )
            w2_out = ttnn.add(w2_out, self.w2_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        original_shape = w2_out.shape
        return ttnn.reshape(
            w2_out, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
