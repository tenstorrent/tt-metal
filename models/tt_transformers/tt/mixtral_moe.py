# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_reduce
from ttnn import ReplicateTensorToMesh, ShardTensorToMesh


class TtMoeLayer(LightweightModule):
    def __init__(self, mesh_device, state_dict, experts, args, layer_num: int, dtype, tt_ccl):
        super().__init__()
        self.mesh_device = mesh_device
        self.experts = experts
        self.args = args
        self.dtype = dtype
        self.model_config = args.get_model_config()
        self.tile_size = args.tile_size
        assert self.tile_size == 32, "tile size must be 32"
        self.num_devices = args.num_devices
        assert self.num_devices == 8, "num devices must be 8 for Mixtral MoE"
        self.tt_ccl = tt_ccl

        gate_name = f"layers.{layer_num}.block_sparse_moe.gate.weight"
        if args.dummy_weights:
            cache_name = None
        else:
            cache_name = args.weight_cache_path(dtype) / (gate_name + "_multidevice_repadded")

        # make the index of the expert on each devices equal to zero
        gates_tensor = (
            torch.nn.functional.pad(state_dict[gate_name].permute(1, 0), (0, 56), "constant", 0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        gates_tensor_list = []
        for dev in range(self.num_devices):
            i, j = 0, dev
            gates_tensor_dev = gates_tensor.clone()
            gates_tensor_dev[:, :, :, [i, j]] = gates_tensor_dev[:, :, :, [j, i]]
            gates_tensor_list.append(gates_tensor_dev)

        self.gates_H8 = ttnn.as_tensor(
            torch.cat(gates_tensor_list, dim=1),
            dtype=ttnn.bfloat16,
            layout=self.model_config["GATE_W_LAYOUT_TILE"],
            memory_config=self.model_config["GATE_WEIGHTS_MEMCFG"],
            cache_file_name=cache_name,
            device=self.mesh_device,
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=1),
        )

        self.compute_kernel = self.args.compute_kernel_config_lofi

        self.compute_kernel_reduce = self.args.compute_kernel_config_hifi2

        top8_mask = torch.full((1, 1, 1, 64), fill_value=torch.finfo(torch.float).min)
        top8_mask[:, :, :, :8] = 0.0
        self.top8_mask_11B_64 = ttnn.from_torch(
            top8_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        self.top8_mask_11B_64 = ttnn.sum(self.top8_mask_11B_64, dim=2)

        top2_mask = torch.full((1, 1, 1, 32), fill_value=torch.finfo(torch.float).min)
        top2_mask[:, :, :, :2] = 0.0
        self.top2_mask_11BB = ttnn.from_torch(
            top2_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        self.top2_mask_11BB = ttnn.sum(self.top2_mask_11BB, dim=2)

        reduce_mask_torch = torch.zeros(1, 1, self.tile_size, self.tile_size * 8)
        for i in range(self.tile_size):
            reduce_mask_torch[:, :, i, range(i, self.tile_size * 8, self.tile_size)] = 1
        self.reduce_mask = ttnn.from_torch(
            reduce_mask_torch,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )

    def forward(self, inputs, mode="decode"):
        """
        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B : batch_size (32)
        H : dim (4096)
        S : seq len
        """
        input_i_1SBH = inputs
        expert_i_HH = self.experts
        # get logits for the experts
        gate_logits_1SB8 = ttnn.matmul(
            input_i_1SBH,
            self.gates_H8,
            memory_config=self.model_config["GATE_MM_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_config["MIXTRAL_GATE_MM_OUTPUT_KERNEL_CONFIG"],
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=ttnn.bfloat16,
        )
        # get weights for top-2 experts -- masking out everything except the 8 experts (needed because top-k works with a min input of size 64)
        gate_logits_1SB8 = ttnn.add(gate_logits_1SB8, self.top8_mask_11B_64)

        if mode == "decode":
            weights_1SB1 = ttnn.moe(gate_logits_1SB8, self.top8_mask_11B_64, self.top2_mask_11BB, 32)
        else:
            topk_values, topk_indices = ttnn.topk(gate_logits_1SB8, 32)
            topk_values = ttnn.add(topk_values, self.top2_mask_11BB)
            mask_B2 = ttnn.eqz(topk_indices)
            mask_B2 = ttnn.typecast(mask_B2, dtype=ttnn.bfloat16)
            weights_1SB1 = ttnn.sum(ttnn.softmax(topk_values, dim=-1) * mask_B2, dim=3)
            topk_values.deallocate(True)
            topk_indices.deallocate(True)
            mask_B2.deallocate(True)

        gate_logits_1SB8.deallocate()
        # MLP and masking
        weights = expert_i_HH(input_i_1SBH, mode=mode)

        if mode == "prefill":
            weights_1SB1 = ttnn.unsqueeze(weights_1SB1, dim=3)
            results_11BH = ttnn.mul(weights, weights_1SB1)
        else:
            results_11BH = ttnn.mul(weights, weights_1SB1)

        weights.deallocate(True)
        weights_1SB1.deallocate(True)

        seq_len = results_11BH.shape[-2]

        if seq_len >= 2048 and mode == "decode":  # Reshape back to intended shape
            results_11BH = ttnn.reshape(results_11BH, [1, 1, seq_len, self.args.dim])

        # All gather
        output = tt_all_reduce(
            results_11BH,
            self.mesh_device,
            tt_ccl=self.tt_ccl,
            cluster_axis=0,
            dim=3,
            num_reduce_scatter_links=self.args.num_reduce_scatter_links,
            num_all_gather_links=self.args.num_all_gather_links,
            sharded=(mode == "decode"),
            memory_config=(results_11BH.memory_config() if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG),
            dtype=self.args.ccl_dtype,
            use_composite=False,
            topology=self.args.ccl_topology(),
        )
        # Ensure dim 0 and 1 are 1
        original_shape = output.shape
        output = ttnn.reshape(
            output, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

        if mode == "decode":  # Decode mode
            results_11BH.deallocate(True)
            output = ttnn.to_memory_config(
                output,
                self.model_config["DECODE_RESIDUAL_MEMCFG"],
            )

        output = ttnn.to_memory_config(
            output,
            memory_config=ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.DRAM
            ),
        )

        return output
