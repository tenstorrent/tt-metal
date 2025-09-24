import torch

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name


class Experts:
    def __init__(self, mesh_device, hf_config, state_dict, ccl_manager, dtype=ttnn.bfloat16, tensor_cache_path=None):
        self.intermediate_size = hf_config.intermediate_size
        self.num_experts = hf_config.num_local_experts
        self.hidden_size = hf_config.hidden_size
        self.expert_dim = self.intermediate_size
        self.ccl_manager = ccl_manager
        self.mesh_device = mesh_device
        self.num_experts_per_tok = hf_config.num_experts_per_tok

        self.intermediate_size_per_device = self.intermediate_size // mesh_device.shape[1]

        gate_proj = state_dict["gate_up_proj"][..., ::2].reshape(1, self.num_experts, self.hidden_size, self.expert_dim)
        up_proj = state_dict["gate_up_proj"][..., 1::2].reshape(1, self.num_experts, self.hidden_size, self.expert_dim)
        gate_proj_bias = state_dict["gate_up_proj_bias"][..., ::2].reshape(1, self.num_experts, 1, self.expert_dim)
        up_proj_bias = state_dict["gate_up_proj_bias"][..., 1::2].reshape(1, self.num_experts, 1, self.expert_dim)
        col_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
        row_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-2)
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
        if mesh_device.shape[1] > 1:
            down_proj_bias = torch.cat(
                [down_proj_bias] + [torch.zeros_like(down_proj_bias)] * (mesh_device.shape[1] - 1), dim=-1
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

        self.alpha = 1.702
        self.limit = 7.0

    def run_dense_experts(self, hidden_states, routing_weights):
        batch_size = hidden_states.shape[0]
        assert batch_size == 1, "batch_size must be 1"
        seq_len = hidden_states.shape[1]
        hidden_states = ttnn.reshape(
            hidden_states, (batch_size, 1, seq_len, self.hidden_size)
        )  # unsqueeze a dim for expert broadcast
        hidden_states_repeated = ttnn.repeat(hidden_states, repeat_dims=(1, self.num_experts, 1, 1))
        # hidden_states.deallocate(True)
        # print("num_experts", self.num_experts)
        # print("hidden_states_repeated", hidden_states_repeated.shape)
        # print("self.gate_proj", self.gate_proj.shape)

        gate_unclamped = ttnn.matmul(hidden_states_repeated, self.gate_proj)  # , dtype=ttnn.bfloat8_b)
        gate_unclamped = ttnn.add(gate_unclamped, self.gate_proj_bias, output_tensor=gate_unclamped)
        up_unclamped = ttnn.matmul(hidden_states_repeated, self.up_proj)  # , dtype=ttnn.bfloat8_b)
        up_unclamped = ttnn.add(up_unclamped, self.up_proj_bias, output_tensor=up_unclamped)
        hidden_states_repeated.deallocate(True)

        gate = ttnn.clamp(gate_unclamped, min=None, max=self.limit)
        up = ttnn.clamp(up_unclamped, min=-self.limit, max=self.limit)
        gate_unclamped.deallocate(True)
        up_unclamped.deallocate(True)
        glu = ttnn.mul(gate, ttnn.sigmoid(gate * self.alpha), output_tensor=gate)
        next_states = ttnn.matmul(((up + 1) * glu), self.down_proj, dtype=ttnn.bfloat16)
        next_states = ttnn.add(next_states, self.down_proj_bias, output_tensor=next_states)
        # gate.deallocate(True)
        up.deallocate(True)
        glu.deallocate(True)
        routing_weights = ttnn.permute(routing_weights, (1, 0))
        routing_weights = ttnn.reshape(routing_weights, (batch_size, self.num_experts, seq_len, 1))
        next_states = ttnn.mul(next_states, routing_weights, output_tensor=next_states)
        routing_weights.deallocate(True)
        next_states = ttnn.sum(next_states, dim=1, keepdim=True)

        if self.mesh_device.shape[1] > 1:
            # AllReduce
            if next_states.shape[-2] >= 32 and self.mesh_device.shape[1] == 8:
                next_states = ttnn.pad(next_states, [(0, 0), (0, 0), (0, 0), (0, 192)], 0)
            next_states_scattered = ttnn.experimental.reduce_scatter_minimal_async(
                next_states,
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_rs_ping_pong_semaphore(),
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.ccl_manager.topology,
                cluster_axis=1,
                barrier_semaphore=self.ccl_manager.get_barrier_semaphore(),
            )
            next_states = ttnn.experimental.all_gather_async(
                next_states_scattered,
                dim=3,
                cluster_axis=1,
                mesh_device=self.ccl_manager.mesh_device,
                topology=self.ccl_manager.topology,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=self.ccl_manager.get_barrier_semaphore(),
            )
            if next_states.shape[-2] >= 32 and self.mesh_device.shape[1] == 8:
                next_states = next_states[:, :, :, : self.hidden_size]
            next_states = ttnn.reshape(next_states, (batch_size, seq_len, self.hidden_size))

        return next_states

    def run_sparse_experts(self, hidden_states, routing_weights):
        batch_size = hidden_states.shape[0]
        assert batch_size == 1, "batch_size must be 1"
        seq_len = hidden_states.shape[1]
        hidden_states_4D = ttnn.unsqueeze_to_4D(hidden_states)

        routing_weights_rm = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing_weights), ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([32, 32])
        # routing_weights_rm = ttnn.transpose(routing_weights_rm, 1, 3)
        # hidden_states_4D = ttnn.transpose(hidden_states_4D, 1, 2)
        # print("hidden_states_4D", hidden_states_4D.shape)
        # print("self.gate_proj", self.gate_proj.shape)
        # print("routing_weights_rm", routing_weights_rm.shape)
        # print("self.num_experts_per_tok", self.num_experts_per_tok)

        # >>> # Sparse matmul for 64 batch, 128 sequence, 512 hidden dimensions, 8 experts
        # >>> tokens = ttnn.ones([1, 64, 128, 512]) [1, 1, 1024, 2880]
        # >>> expert_weights = ttnn.ones([1, 8, 512, 512]) [1, 128, 2880, 360]
        # >>> # Create sparsity bitmask
        # >>> sparsity_bitmask = torch.zeros([1, 64, 128, 8])     [1, 1, 1024, 128]
        # [batch_size, seq_len, 1, hidden_size]
        # ttnn.synchronize_device(self.mesh_device)
        gate = ttnn.sparse_matmul(
            hidden_states_4D,
            self.gate_proj,
            sparsity=routing_weights_rm,
            nnz=self.num_experts_per_tok,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
        )
        # ttnn.synchronize_device(self.mesh_device)
        # ("done sparse matmul")
        gate = ttnn.reshape(gate, (batch_size, self.num_experts, seq_len, self.intermediate_size_per_device))
        gate = ttnn.add(gate, self.gate_proj_bias, output_tensor=gate)
        gate = ttnn.clamp(gate, min=None, max=self.limit)

        up = ttnn.sparse_matmul(
            hidden_states_4D,
            self.up_proj,
            sparsity=routing_weights_rm,
            nnz=self.num_experts_per_tok,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
        )
        # ttnn.deallocate(hidden_states_4D)
        up = ttnn.reshape(up, (batch_size, self.num_experts, seq_len, self.intermediate_size_per_device))
        up = ttnn.add(up, self.up_proj_bias, output_tensor=up)
        up = ttnn.clamp(up, min=-self.limit, max=self.limit)

        glu = gate * ttnn.sigmoid(gate * self.alpha)
        down_in0 = (up + 1) * glu
        ttnn.deallocate(glu)
        ttnn.deallocate(up)
        ttnn.deallocate(gate)
        down_in0 = ttnn.reshape(down_in0, (1, self.num_experts, seq_len, self.intermediate_size_per_device))

        down = ttnn.sparse_matmul(
            down_in0,
            self.down_proj,
            sparsity=routing_weights_rm,
            nnz=self.num_experts_per_tok,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
        )
        # ttnn.deallocate(down_in0)
        # ttnn.deallocate(routing_weights_rm)

        next_states = (
            ttnn.reshape(down, (batch_size, self.num_experts, seq_len, self.hidden_size)) + self.down_proj_bias
        )

        routing_weights = ttnn.permute(routing_weights, (1, 0))
        routing_weights = ttnn.reshape(routing_weights, (batch_size, self.num_experts, seq_len, 1))

        next_states = ttnn.mul(next_states, routing_weights, output_tensor=next_states)
        next_states = ttnn.sum(next_states, dim=1, keepdim=True)

        if self.mesh_device.shape[1] > 1:
            # AllReduce
            next_states_scattered = ttnn.experimental.reduce_scatter_minimal_async(
                next_states,
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_rs_ping_pong_semaphore(),
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.ccl_manager.topology,
                cluster_axis=1,
                barrier_semaphore=self.ccl_manager.get_barrier_semaphore(),
            )
            next_states = ttnn.experimental.all_gather_async(
                next_states_scattered,
                dim=3,
                cluster_axis=1,
                mesh_device=self.ccl_manager.mesh_device,
                topology=self.ccl_manager.topology,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=self.ccl_manager.get_barrier_semaphore(),
            )
            next_states = ttnn.reshape(next_states, (batch_size, seq_len, self.hidden_size))

        return next_states

    def __call__(self, hidden_states, routing_weights):
        # If decode mode, we use sparse experts for better performance else use dense experts
        if hidden_states.shape[-2] == 1:
            return self.run_sparse_experts(hidden_states, routing_weights)
        else:
            return self.run_dense_experts(hidden_states, routing_weights)
        return self.run_dense_experts(hidden_states, routing_weights)


class SparseExperts(Experts):
    def __init__(self, mesh_device, hf_config, state_dict, ccl_manager, dtype=ttnn.bfloat16, tensor_cache_path=None):
        super().__init__(mesh_device, hf_config, state_dict, ccl_manager, dtype, tensor_cache_path)
        self.num_experts_per_tok = hf_config.num_experts_per_tok

    def __call__(self, hidden_states, routing_weights):
        batch_size = hidden_states.shape[0]
        assert batch_size == 1, "batch_size must be 1"
        seq_len = hidden_states.shape[1]
        hidden_states_4D = ttnn.unsqueeze_to_4D(hidden_states)

        routing_weights_rm = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing_weights), ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([32, 32])

        # [batch_size, seq_len, 1, hidden_size]
        gate = ttnn.sparse_matmul(
            hidden_states_4D,
            self.gate_proj,
            sparsity=routing_weights_rm,
            nnz=self.num_experts_per_tok,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
        )
        gate = (
            ttnn.reshape(gate, (batch_size, self.num_experts, seq_len, self.intermediate_size_per_device))
            + self.gate_proj_bias
        )
        gate = ttnn.clamp(gate, min=None, max=self.limit)

        up = ttnn.sparse_matmul(
            hidden_states_4D,
            self.up_proj,
            sparsity=routing_weights_rm,
            nnz=self.num_experts_per_tok,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
        )
        # ttnn.deallocate(hidden_states_4D)

        up = (
            ttnn.reshape(up, (batch_size, self.num_experts, seq_len, self.intermediate_size_per_device))
            + self.up_proj_bias
        )
        up = ttnn.clamp(up, min=-self.limit, max=self.limit)

        glu = gate * ttnn.sigmoid(gate * self.alpha)
        down_in0 = (up + 1) * glu
        ttnn.deallocate(glu)
        ttnn.deallocate(up)
        ttnn.deallocate(gate)
        down_in0 = ttnn.reshape(down_in0, (1, self.num_experts, seq_len, self.intermediate_size_per_device))

        down = ttnn.sparse_matmul(
            down_in0,
            self.down_proj,
            sparsity=routing_weights_rm,
            nnz=self.num_experts_per_tok,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
        )
        ttnn.deallocate(down_in0)
        # ttnn.deallocate(routing_weights_rm)

        next_states = (
            ttnn.reshape(down, (batch_size, self.num_experts, seq_len, self.hidden_size)) + self.down_proj_bias
        )

        routing_weights = ttnn.permute(routing_weights, (1, 0))
        routing_weights = ttnn.reshape(routing_weights, (batch_size, self.num_experts, seq_len, 1))

        next_states = ttnn.mul(next_states, routing_weights, output_tensor=next_states)
        next_states = ttnn.sum(next_states, dim=1, keepdim=True)

        if self.mesh_device.shape[1] > 1:
            # AllReduce
            next_states_scattered = ttnn.experimental.reduce_scatter_minimal_async(
                next_states,
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_rs_ping_pong_semaphore(),
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.ccl_manager.topology,
                cluster_axis=1,
                barrier_semaphore=self.ccl_manager.get_barrier_semaphore(),
            )
            next_states.deallocate(True)
            next_states = ttnn.experimental.all_gather_async(
                next_states_scattered,
                dim=3,
                cluster_axis=1,
                mesh_device=self.ccl_manager.mesh_device,
                topology=self.ccl_manager.topology,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=self.ccl_manager.get_barrier_semaphore(),
            )
            next_states_scattered.deallocate(True)
            next_states = ttnn.reshape(next_states, (batch_size, seq_len, self.hidden_size))

        return next_states
