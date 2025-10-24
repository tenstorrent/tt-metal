# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.demos.gpt_oss.config import MeshConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name


class Experts:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=None,
    ):
        self.intermediate_size = hf_config.intermediate_size
        self.num_experts = hf_config.num_local_experts
        self.hidden_size = hf_config.hidden_size
        self.expert_dim = self.intermediate_size
        self.ccl_manager = ccl_manager
        self.mesh_device = mesh_device
        self.num_experts_per_tok = hf_config.num_experts_per_tok

        # Use MeshConfig for clean parallelization
        self.mesh_config = mesh_config or MeshConfig(
            mesh_device.shape, tp=mesh_device.shape[1], ep=mesh_device.shape[0]
        )
        self.intermediate_size_per_device = self.mesh_config.shard_size(self.intermediate_size)

        gate_proj = state_dict["gate_up_proj"][..., ::2].reshape(1, self.num_experts, self.hidden_size, self.expert_dim)
        up_proj = state_dict["gate_up_proj"][..., 1::2].reshape(1, self.num_experts, self.hidden_size, self.expert_dim)
        gate_proj_bias = state_dict["gate_up_proj_bias"][..., ::2].reshape(1, self.num_experts, 1, self.expert_dim)
        up_proj_bias = state_dict["gate_up_proj_bias"][..., 1::2].reshape(1, self.num_experts, 1, self.expert_dim)

        # Clean mesh mapping using MeshConfig
        col_mesh_mapper = self.mesh_config.column_parallel(mesh_device)
        row_mesh_mapper = self.mesh_config.row_parallel(mesh_device)
        dtype = ttnn.bfloat4_b
        self.gate_proj = ttnn.as_tensor(
            gate_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.up_proj = ttnn.as_tensor(
            up_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"up_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.gate_proj_bias = ttnn.as_tensor(
            gate_proj_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_proj_bias"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.up_proj_bias = ttnn.as_tensor(
            up_proj_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"up_proj_bias"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        down_proj = state_dict["down_proj"].reshape(1, self.num_experts, self.expert_dim, self.hidden_size)
        down_proj_bias = state_dict["down_proj_bias"].reshape(1, self.num_experts, 1, self.hidden_size)
        self.down_proj = ttnn.as_tensor(
            down_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=row_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"down_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Row-parallel bias must not be replicated. Extend it with zeros for TP devices.
        if self.mesh_config.tp > 1:
            down_proj_bias = torch.cat(
                [down_proj_bias] + [torch.zeros_like(down_proj_bias)] * (self.mesh_config.tp - 1), dim=-1
            )
        self.down_proj_bias = ttnn.as_tensor(
            down_proj_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"down_proj_bias"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.alpha = 1.702  # from https://github.com/huggingface/transformers/blob/b4067472aee9b566237091dbcd3659dd2ce92004/src/transformers/models/gpt_oss/modular_gpt_oss.py#L77
        self.limit = hf_config.swiglu_limit

        tokens_per_ep = self.num_experts // self.mesh_config.ep
        sparsity = torch.zeros(1, 1, self.mesh_config.ep, self.num_experts)
        for i in range(self.mesh_config.ep):
            sparsity[:, :, i, i * tokens_per_ep : (i + 1) * tokens_per_ep] = torch.ones(1, 1, 1, tokens_per_ep)
        self.prefill_sparsity = ttnn.from_torch(
            sparsity,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                dims=(-2, None), mesh_shape=self.mesh_device.shape, mesh_device=self.mesh_device
            ),
        )

        self.sparse_matmul_program_config = (
            lambda core_x, core_y, m, n: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=1,
                per_core_M=max(32, m) // 32,
                per_core_N=int(math.ceil(n / 32)) // (core_x * core_y),
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
            )
        )
        self.batched_sparse_matmul_program_config = (
            lambda core_x, core_y, m, n: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=1,
                per_core_M=max(32, m) // 32,
                per_core_N=int(math.ceil(n / 32)) // (core_x * core_y),
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
            )
        )

    def __call__(self, hidden_states, routing_weights):
        batch_size = hidden_states.shape[0]
        assert batch_size == 1, "batch_size must be 1, we only support batch size 1 for now"
        seq_len = hidden_states.shape[1]
        hidden_states_4D = ttnn.unsqueeze_to_4D(hidden_states)
        if seq_len > 1:
            TILE_SIZE = 32
            hidden_states_4D = ttnn.reshape(hidden_states_4D, (1, seq_len // TILE_SIZE, TILE_SIZE, self.hidden_size))
            group_size = seq_len // TILE_SIZE
        sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing_weights), ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([32, 32])

        if seq_len > 1:
            sparsity = ttnn.repeat(self.prefill_sparsity, (1, 1, group_size, 1))

        if self.mesh_config.ep > 1 and seq_len == 1:
            routing_weights_rm_torch = ttnn.to_torch(ttnn.get_device_tensors(sparsity)[0]).reshape(-1)
            # find the indices of the non-zero values in routing_weights_rm_torch
            non_zero_indices = torch.nonzero(routing_weights_rm_torch)
            # create tnsirs which contain only 1 of the non-zero values
            non_zero_indices = non_zero_indices.reshape(-1)
            routing_weights_rm_torch_list = []
            for i in range(non_zero_indices.shape[0]):
                routing_weights_rm_torch_0 = torch.zeros_like(routing_weights_rm_torch)
                routing_weights_rm_torch_0[non_zero_indices[i]] = routing_weights_rm_torch[non_zero_indices[i]]
                routing_weights_rm_torch_list.append(routing_weights_rm_torch_0.reshape(1, 1, 1, self.num_experts))
            routing_weights_rm_torch = torch.cat(routing_weights_rm_torch_list, dim=2)
            sparsity = ttnn.from_torch(
                routing_weights_rm_torch,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    dims=(-2, None), mesh_shape=self.mesh_device.shape, mesh_device=self.mesh_device
                ),
            )
            routing_weights = ttnn.from_torch(
                routing_weights_rm_torch.reshape(4, -1),
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    dims=(-2, None), mesh_shape=self.mesh_device.shape, mesh_device=self.mesh_device
                ),
            )

        num_experts_per_tok = (
            (self.num_experts // self.mesh_config.ep) * group_size
            if seq_len > 1
            else self.num_experts_per_tok // self.mesh_config.ep
        )
        program_config = self.sparse_matmul_program_config(3, 4, hidden_states_4D.shape[2], self.gate_proj.shape[3])

        gate = ttnn.sparse_matmul(
            hidden_states_4D,
            self.gate_proj,
            sparsity=sparsity,
            nnz=num_experts_per_tok,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=program_config,
            dtype=ttnn.bfloat8_b if seq_len > 1 else ttnn.bfloat16,
        )

        if seq_len > 1:
            gate_transposed = ttnn.transpose(gate, 1, 3)
            gate.deallocate(True)
            gate = gate_transposed

        gate = ttnn.reshape(gate, (batch_size, self.num_experts, seq_len, self.intermediate_size_per_device))
        gate = ttnn.add(gate, self.gate_proj_bias, output_tensor=gate)
        gate_clamped = ttnn.clamp(gate, min=None, max=self.limit)
        gate.deallocate(True)
        gate = gate_clamped

        up = ttnn.sparse_matmul(
            hidden_states_4D,
            self.up_proj,
            sparsity=sparsity,
            nnz=num_experts_per_tok,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=program_config,
            dtype=ttnn.bfloat8_b if seq_len > 1 else ttnn.bfloat16,
        )
        hidden_states_4D.deallocate(True)
        if seq_len > 1:
            up_transposed = ttnn.transpose(up, 1, 3)
            up.deallocate(True)
            up = up_transposed
        up = ttnn.reshape(up, (batch_size, self.num_experts, seq_len, self.intermediate_size_per_device))
        up = ttnn.add(up, self.up_proj_bias, output_tensor=up)
        up_clamped = ttnn.clamp(up, min=-self.limit, max=self.limit)
        up.deallocate(True)
        up = up_clamped

        gate = ttnn.mul(gate, self.alpha, output_tensor=gate)
        gate_sigmoid = ttnn.sigmoid(gate)
        glu = ttnn.mul(gate, gate_sigmoid, output_tensor=gate)
        gate_sigmoid.deallocate(True)
        up = ttnn.add(up, 1, output_tensor=up)
        down_in0 = ttnn.mul(up, glu, output_tensor=up)
        ttnn.deallocate(glu)
        down_in0 = ttnn.reshape(down_in0, (1, self.num_experts, seq_len, self.intermediate_size_per_device))
        if seq_len > 1:
            # down_in0 = ttnn.reshape(down_in0, (1, self.num_experts, group_size, seq_len//group_size, self.intermediate_size_per_device))
            # down_in0 = ttnn.transpose(down_in0, 1, 3)
            # down_in0 = ttnn.reshape(down_in0, (1, self.num_experts, seq_len, self.intermediate_size_per_device))
            sparsity = self.prefill_sparsity
            num_experts_per_tok = self.num_experts // self.mesh_config.ep
            routing_weights = ttnn.mul(
                routing_weights,
                ttnn.reshape(self.prefill_sparsity, (1, self.num_experts)),
                output_tensor=routing_weights,
            )

        routing_weights_transposed = ttnn.permute(routing_weights, (1, 0))
        routing_weights.deallocate(True)
        routing_weights = routing_weights_transposed
        routing_weights = ttnn.reshape(routing_weights, (batch_size, self.num_experts, seq_len, 1))

        SPLIT_SIZE = 2048
        if seq_len > SPLIT_SIZE:
            down_in0_list = ttnn.split(down_in0, SPLIT_SIZE, dim=2)
            down_in0.deallocate(True)
            routing_weights_list = ttnn.split(routing_weights, SPLIT_SIZE, dim=2)
            routing_weights.deallocate(True)
        else:
            down_in0_list = [down_in0]
            routing_weights_list = [routing_weights]

        next_states_reduced_list = []
        for i, down_in0 in enumerate(down_in0_list):
            down = ttnn.sparse_matmul(
                down_in0,
                self.down_proj,
                sparsity=sparsity,
                nnz=num_experts_per_tok,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                is_input_a_sparse=True,
                program_config=self.batched_sparse_matmul_program_config(
                    5, 6, down_in0.shape[2], self.down_proj.shape[-1]
                ),
                dtype=ttnn.bfloat8_b if seq_len > 1 else ttnn.bfloat16,
            )
            down_in0.deallocate(True)
            next_states = ttnn.reshape(
                down,
                (batch_size, self.num_experts, (seq_len if seq_len < SPLIT_SIZE else SPLIT_SIZE), self.hidden_size),
            )
            next_states = ttnn.add(next_states, self.down_proj_bias, output_tensor=next_states)

            next_states = ttnn.mul(next_states, routing_weights_list[i], output_tensor=next_states)
            next_states_reduced = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(next_states, dims=[1]))
            next_states.deallocate(True)
            next_states_reduced_list.append(next_states_reduced)
            routing_weights_list[i].deallocate(True)

        next_states = ttnn.concat(next_states_reduced_list, dim=2)

        # EP communication
        if self.mesh_config.ep > 1:
            next_states_allreduced = self.mesh_config.allreduce(
                next_states, self.ccl_manager, axis=self.mesh_config.ep_axis
            )
            next_states.deallocate(True)
            next_states = next_states_allreduced
        # TP communication
        if next_states.dtype != ttnn.bfloat16:
            next_states_16 = ttnn.typecast(next_states, ttnn.bfloat16)
            ttnn.deallocate(next_states)
        else:
            next_states_16 = next_states
        next_states = self.mesh_config.allreduce(
            next_states_16,
            self.ccl_manager,
            pad_size=192 if self.mesh_config.tp == 8 else 0,
            axis=self.mesh_config.tp_axis,
        )

        next_states_16.deallocate(True)

        next_states = ttnn.reshape(
            next_states, (batch_size, seq_len, self.hidden_size), (batch_size, max(32, seq_len), self.hidden_size)
        )
        return next_states
