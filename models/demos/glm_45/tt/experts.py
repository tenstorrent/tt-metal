# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import torch

import ttnn
from models.demos.glm_45.config import MeshConfig
from models.demos.glm_45.utils.general_utils import get_cache_file_name


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
        shared_state_dict=None,
    ):
        # GLM MoE uses moe_intermediate_size; fall back to intermediate_size if not present
        moe_dim = getattr(hf_config, "moe_intermediate_size", None)
        self.intermediate_size = moe_dim if moe_dim is not None else hf_config.intermediate_size
        self.num_experts = hf_config.n_routed_experts
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

        # Build aggregated tensors: gate_up_proj (packed gate/up interleaved), optional bias, and down_proj(+bias)
        if all(k in state_dict for k in ["gate_up_proj", "down_proj"]):
            # Unpack packed tensors; some GLM variants pack as contiguous halves: [gate | up]
            packed = state_dict["gate_up_proj"]  # [E, H, 2*I] expected
            I = self.expert_dim
            gate_proj = packed[..., :I].reshape(self.num_experts, self.hidden_size, self.expert_dim)
            up_proj = packed[..., I:].reshape(self.num_experts, self.hidden_size, self.expert_dim)
            gate_proj_bias = None
            up_proj_bias = None
            if "gate_up_proj_bias" in state_dict:
                packed_b = state_dict["gate_up_proj_bias"]  # [E, 2*I]
                gate_proj_bias = packed_b[..., :I].reshape(self.num_experts, 1, self.expert_dim)
                up_proj_bias = packed_b[..., I:].reshape(self.num_experts, 1, self.expert_dim)
            down_proj = state_dict["down_proj"].reshape(self.num_experts, self.expert_dim, self.hidden_size)
            down_proj_bias = state_dict.get("down_proj_bias")
            if down_proj_bias is not None:
                down_proj_bias = down_proj_bias.reshape(self.num_experts, 1, self.hidden_size)
        else:
            # Build stacked tensors from per-expert weights
            gate_proj = torch.empty(self.num_experts, self.hidden_size, self.expert_dim, dtype=torch.bfloat16)
            up_proj = torch.empty(self.num_experts, self.hidden_size, self.expert_dim, dtype=torch.bfloat16)
            down_proj = torch.empty(self.num_experts, self.expert_dim, self.hidden_size, dtype=torch.bfloat16)
            gate_proj_bias = None
            up_proj_bias = None
            down_proj_bias = None
            for i in range(self.num_experts):
                gp = state_dict.get(f"{i}.gate_proj.weight")
                up = state_dict.get(f"{i}.up_proj.weight")
                dp = state_dict.get(f"{i}.down_proj.weight")
                if gp is None or up is None or dp is None:
                    raise ValueError(
                        f"Missing per-expert weights for expert {i}. Required: "
                        f"experts.{i}.gate_proj.weight, experts.{i}.up_proj.weight, experts.{i}.down_proj.weight"
                    )
                # Load as [E, H, I] for gate/up and [E, I, H] for down
                gate_proj[i] = gp.transpose(0, 1).to(torch.bfloat16)
                up_proj[i] = up.transpose(0, 1).to(torch.bfloat16)
                down_proj[i] = dp.transpose(0, 1).to(torch.bfloat16)

        # Mesh mapping using MeshConfig
        col_mesh_mapper = self.mesh_config.column_parallel(mesh_device)
        row_mesh_mapper = self.mesh_config.row_parallel(mesh_device)

        # Use higher precision weights to improve numerical match (especially for 1x1 case)
        weight_dtype = ttnn.bfloat16
        self.gate_proj = ttnn.as_tensor(
            gate_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "gate_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.up_proj = ttnn.as_tensor(
            up_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "up_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.gate_proj_bias = (
            ttnn.as_tensor(
                gate_proj_bias,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=weight_dtype,
                mesh_mapper=col_mesh_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "gate_proj_bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if gate_proj_bias is not None
            else None
        )
        self.up_proj_bias = (
            ttnn.as_tensor(
                up_proj_bias,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=weight_dtype,
                mesh_mapper=col_mesh_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "up_proj_bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if up_proj_bias is not None
            else None
        )

        self.down_proj = ttnn.as_tensor(
            down_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=row_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "down_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Ensure there is no leading singleton dimension in weights/biases
        def _squeeze_leading1(t):
            if t is None:
                return None
            shp_list = list(t.shape)
            if len(shp_list) > 0 and int(shp_list[0]) == 1:
                new_shape = tuple(int(x) for x in shp_list[1:])
                if len(new_shape) == 0:
                    return t
                return ttnn.reshape(t, new_shape)
            return t

        # Row-parallel bias must not be replicated. Extend with zeros for TP devices.
        if down_proj_bias is not None:
            down_bias = down_proj_bias
            if self.mesh_config.tp > 1:
                down_bias = torch.cat([down_bias] + [torch.zeros_like(down_bias)] * (self.mesh_config.tp - 1), dim=-1)
            self.down_proj_bias = ttnn.as_tensor(
                down_bias,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=weight_dtype,
                mesh_mapper=col_mesh_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "down_proj_bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.down_proj_bias = None

        # Final sanity: ensure no leading singleton dims on weights/biases
        self.gate_proj = _squeeze_leading1(self.gate_proj)
        self.up_proj = _squeeze_leading1(self.up_proj)
        self.down_proj = _squeeze_leading1(self.down_proj)
        self.gate_proj_bias = _squeeze_leading1(self.gate_proj_bias)
        self.up_proj_bias = _squeeze_leading1(self.up_proj_bias)
        self.down_proj_bias = _squeeze_leading1(self.down_proj_bias)

        # Optional shared expert MLP (dense path)
        # Expect weights under shared_experts.(gate_proj|up_proj|down_proj).weight
        # Shapes follow DenseGLU: gate/up (H, I_shared), down (I_shared, H)
        self.has_shared = False
        n_shared = getattr(hf_config, "n_shared_experts", 0) or 0
        if n_shared > 0 and shared_state_dict is not None and len(shared_state_dict) > 0:
            required_shared = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
            missing_shared = [k for k in required_shared if k not in shared_state_dict]
            if missing_shared:
                raise ValueError(
                    f"Missing shared_experts weights: {missing_shared}. n_shared_experts={n_shared} requires them."
                )

            # Transpose HF weights to (in_dim, out_dim)
            se_gate = shared_state_dict["gate_proj.weight"].transpose(0, 1)
            se_up = shared_state_dict["up_proj.weight"].transpose(0, 1)
            se_down = shared_state_dict["down_proj.weight"].transpose(0, 1)

            # Optional biases
            se_gate_b = shared_state_dict.get("gate_proj.bias")
            se_up_b = shared_state_dict.get("up_proj.bias")
            se_down_b = shared_state_dict.get("down_proj.bias")

            # Map weights like other experts: col-parallel for gate/up, row-parallel for down
            self.shared_gate_w = ttnn.as_tensor(
                se_gate,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=weight_dtype,
                mesh_mapper=col_mesh_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "shared_gate_proj"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.shared_up_w = ttnn.as_tensor(
                se_up,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=weight_dtype,
                mesh_mapper=col_mesh_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "shared_up_proj"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.shared_down_w = ttnn.as_tensor(
                se_down,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=weight_dtype,
                mesh_mapper=row_mesh_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "shared_down_proj"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Bias tensors if present; keep 2D with leading batch dim for broadcast
            self.shared_gate_b = (
                ttnn.as_tensor(
                    se_gate_b.unsqueeze(0),
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=weight_dtype,
                    cache_file_name=get_cache_file_name(tensor_cache_path, "shared_gate_proj_bias"),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if se_gate_b is not None
                else None
            )
            self.shared_up_b = (
                ttnn.as_tensor(
                    se_up_b.unsqueeze(0),
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=weight_dtype,
                    cache_file_name=get_cache_file_name(tensor_cache_path, "shared_up_proj_bias"),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if se_up_b is not None
                else None
            )
            self.shared_down_b = (
                ttnn.as_tensor(
                    se_down_b.unsqueeze(0),
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=weight_dtype,
                    cache_file_name=get_cache_file_name(tensor_cache_path, "shared_down_proj_bias"),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if se_down_b is not None
                else None
            )

            self.has_shared = True

    def __call__(self, hidden_states, routing_weights):
        # hidden_states: [B, S, H] in TILE layout (keep 3D)
        # routing_weights: [B*S, E] (from router), TILE or ROW_MAJOR
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # Pack all tokens across batches into tokens_3d[0] with shape [S_total, 1, H]
        tokens_total = batch_size * seq_len
        tokens_3d = ttnn.reshape(hidden_states, (tokens_total, 1, self.hidden_size))
        print("tokens_3d packed: ", tokens_3d.shape)

        # Convert routing weights to ROW_MAJOR 3D tensor [1, S_total, E]
        rw_2d = (
            routing_weights if len(routing_weights.shape) == 2 else ttnn.reshape(routing_weights, (tokens_total, -1))
        )
        print("rw_2d: ", rw_2d)
        rw_rm = ttnn.to_layout(rw_2d, ttnn.ROW_MAJOR_LAYOUT)
        weights_3d = ttnn.reshape(rw_rm, (1, tokens_total, self.num_experts))
        # Binary mask for expert selection (top-k), used in sparse matmuls to compute only selected experts
        mask = ttnn.gt(weights_3d, 0.0)

        # Program config: mirror smoke3 (1D multicast, in0 multicast)
        output_tile = ttnn.Tile([32, 32])

        down_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=2,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

        # Gate
        print("going into gate: ", tokens_3d.shape, self.gate_proj.shape, weights_3d.shape)
        gate = ttnn.sparse_matmul(
            tokens_3d,
            self.gate_proj,
            sparsity=mask,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            # program_config=pc,
        )

        # Up
        up = ttnn.sparse_matmul(
            tokens_3d,
            self.up_proj,
            sparsity=mask,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
        )
        # Activation: GLM experts use SwiGLU -> silu(gate) * up
        glu = ttnn.silu(gate)
        down_in0 = ttnn.mul(glu, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(glu)
        ttnn.deallocate(up)
        ttnn.deallocate(gate)

        # Down: batched sparse matmul with sparse A
        down = ttnn.sparse_matmul(
            down_in0,
            self.down_proj,
            sparsity=mask,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            program_config=down_pc,
        )
        # Keep down-proj bias disabled here; GLM reference often omits it

        # Bias and combine experts using routing weights
        # down currently shaped like [S_total, E, 1, H_per_tp]; reshape to [B, E, S, H]
        next_states = ttnn.reshape(down, (batch_size, self.num_experts, seq_len, self.hidden_size))
        # Explicitly apply routing weights at combine time: [B,E,S,H] * [B,E,S,1]
        w = ttnn.reshape(rw_rm, (batch_size, seq_len, self.num_experts))  # [B,S,E]
        w = ttnn.transpose(w, 1, 2)  # [B,E,S]
        w = ttnn.unsqueeze(w, -1)  # [B,E,S,1]
        next_states = ttnn.mul(next_states, w, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        next_states = ttnn.sum(next_states, dim=1, keepdim=True)

        # Add shared experts dense path before allreduce (single combined allreduce)
        if getattr(self, "has_shared", False):
            gate_s = ttnn.linear(hidden_states, self.shared_gate_w, bias=self.shared_gate_b)
            up_s = ttnn.linear(hidden_states, self.shared_up_w, bias=self.shared_up_b)
            glu_s = ttnn.silu(gate_s)
            down_in_s = ttnn.mul(glu_s, up_s, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(glu_s)
            ttnn.deallocate(up_s)
            ttnn.deallocate(gate_s)

            shared_out = ttnn.linear(down_in_s, self.shared_down_w, bias=self.shared_down_b)
            ttnn.deallocate(down_in_s)
            # Match next_states pre-allreduce shape: [B, 1, S, H]
            shared_out = ttnn.reshape(shared_out, (batch_size, 1, seq_len, self.hidden_size))
            print("shared_out: ", shared_out)
            next_states = next_states + shared_out * 4

        # EP allreduce then TP allreduce (single path for routed + shared)
        if self.mesh_config.ep > 1:
            next_states = self.mesh_config.allreduce(next_states, self.ccl_manager, axis=self.mesh_config.ep_axis)
        next_states = self.mesh_config.allreduce(
            next_states,
            self.ccl_manager,
            pad_size=192 if self.mesh_config.tp == 8 else 0,
            axis=self.mesh_config.tp_axis,
        )
        next_states = ttnn.reshape(next_states, (batch_size, seq_len, self.hidden_size))
        return next_states
