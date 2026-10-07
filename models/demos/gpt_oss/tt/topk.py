# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Top-K Router implementation for Mixture of Experts (MoE) in GPT-OSS.

This module implements the routing mechanism that selects which experts should
process each token in the MoE architecture. The router uses a learned linear
transformation followed by top-k selection to assign tokens to experts.

"""

import torch

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name

from .fused_decode import auto_matmul_compute_config, matmul_1d_program_config


def topk_router(g, experts_per_token, use_throughput_experts, softmax_compute_config=None):
    typecast_needed = False
    if g.dtype != ttnn.bfloat16:
        g_og = g
        typecast_needed = True
        g = ttnn.typecast(g, dtype=ttnn.bfloat16)

    expert_weights, expert_indices = ttnn.topk(g, k=experts_per_token, dim=-1, sorted=True)
    if typecast_needed:
        g.deallocate(True)
        g = g_og
    if softmax_compute_config is None:
        softmax_compute_config = ttnn.init_device_compute_kernel_config(
            g.device().arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
    expert_weights = ttnn.softmax(
        expert_weights, dim=1, numeric_stable=True, compute_kernel_config=softmax_compute_config
    )
    if use_throughput_experts:
        return expert_indices, expert_weights
    else:
        return expert_indices, ttnn.scatter(ttnn.zeros_like(g), dim=1, index=expert_indices, src=expert_weights)


class TopKRouter:
    def __init__(
        self, mesh_device, hf_config, state_dict, tensor_cache_path=None, indexed_decode=False, ccl_manager=None
    ):
        self.top_k = hf_config.num_experts_per_tok
        self.num_experts = hf_config.num_local_experts
        self.hidden_dim = hf_config.hidden_size
        self.tensor_cache_path = tensor_cache_path
        torch_weight = state_dict["weight"].transpose(0, 1) if state_dict else None
        torch_bias = state_dict["bias"].unsqueeze(0) if state_dict else None
        self.weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bias = ttnn.as_tensor(
            torch_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "bias"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Keep compute_config=None for linear (known quality-safe default)
        # Custom compute configs were previously found to cause quality degradation
        self.compute_config = None

        # Cache softmax compute config (same as what topk_router creates per-call)
        self.softmax_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        if indexed_decode:
            assert ccl_manager is not None, "indexed decode routing shares its gate buffers through the CCL manager"
            self._init_indexed_decode(mesh_device, torch_weight, torch_bias, tensor_cache_path, ccl_manager)

        # Fused op support: matmul + topk + softmax in one kernel
        # The fused kernel uses 4 groups of 3 cores, one per N-tile (32 experts
        # each), so it requires exactly 128 experts and 12 DRAM-aligned cores.
        # Blackhole has only 8 DRAM banks; use the generic router on that architecture.
        # Issue for native 8 bank BH support: https://github.com/tenstorrent/tt-metal/issues/57186
        self.use_fused_op = self.num_experts == 128 and not ttnn.device.is_blackhole(mesh_device)
        self._fused_bias = None
        # Keep the original unsharded bias for fused op initialization
        # (ttnn.as_tensor shards self.bias across the mesh, but the fused op
        # needs the full [1, num_experts] bias replicated on every device)
        if self.use_fused_op and state_dict:
            self._bias_torch = state_dict["bias"].unsqueeze(0).to(torch.bfloat16)
        else:
            self._bias_torch = None

    def _init_indexed_decode(self, mesh_device, torch_weight, torch_bias, tensor_cache_path, ccl_manager):
        # Indexed decode router: ttnn.experimental.deepseek.moe.generalized_moe_gate (top-k + softmax over the
        # selected logits in one op). The op ranks one 256-expert 16x16 face per token, so the router weight and
        # linear bias are zero-padded to 256 columns; the phantom experts carry a -1e9 selection bias so they never
        # rank (the selection bias does not enter the output scores).
        self.decode_width = 256
        self.decode_tokens = 1
        if torch_weight is not None:
            pad = self.decode_width - self.num_experts
            torch_weight_decode = torch.nn.functional.pad(torch_weight, (0, pad))
            torch_bias_decode = torch.nn.functional.pad(torch_bias, (0, pad))
        else:
            torch_weight_decode = torch_bias_decode = None
        self.weight_decode = ttnn.as_tensor(
            torch_weight_decode,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"weight_decode{self.decode_width}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bias_decode = ttnn.as_tensor(
            torch_bias_decode,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"bias_decode{self.decode_width}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ccl_manager.get_decode_gate_buffers(self.num_experts, self.decode_width, self.decode_tokens)
        self.gate_memory_config = gate["memory_config"]
        self.gate_selection_bias = gate["selection_bias"]
        self.gate_expert_ids = gate["expert_ids"]
        self.gate_scores = gate["scores"]
        self.gate_indices = gate["indices"]

    def _init_fused_op(self, device, B):
        """Lazily initialize fused op tensors (bias broadcast + output pre-alloc)."""
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None

        if self._fused_bias is None:
            if self._bias_torch is not None:
                # Use the original unsharded bias (self._bias_torch is [1, num_experts])
                # and broadcast to [B, num_experts] so every tile row has the bias vector.
                bias_bcast = self._bias_torch.expand(B, -1).contiguous()
            else:
                bias_bcast = None
            self._fused_bias = ttnn.as_tensor(
                bias_bcast,
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
                cache_file_name=get_cache_file_name(self.tensor_cache_path, f"fused_bias_B{B}"),
            )

    def __call__(self, hidden_states, use_throughput_experts):
        # Compute actual token count from the tensor volume before reshape,
        # since shape[0] after reshape returns the tile-padded dimension
        # (e.g. 8 tokens padded to 32 in TILE_LAYOUT).
        actual_tokens = hidden_states.volume() // self.hidden_dim
        hidden_states = ttnn.reshape(hidden_states, (-1, self.hidden_dim))

        # Fused op only supports decode mode (B=32, seq_len=1 → shape [32, hidden_dim])
        if self.use_fused_op and actual_tokens == 32 and use_throughput_experts:
            return self._fused_call(hidden_states, use_throughput_experts)

        # Use L1 for decode (small tensors), DRAM for prefill (large sequences)
        is_decode = actual_tokens <= 128
        mem_config = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG
        router_logits = ttnn.linear(
            hidden_states,
            self.weight,
            bias=self.bias,
            memory_config=mem_config,
            compute_kernel_config=self.compute_config,
        )

        expert_indices, expert_weights = topk_router(
            router_logits, self.top_k, use_throughput_experts, self.softmax_compute_config
        )
        ttnn.deallocate(router_logits)
        return expert_indices, expert_weights

    def decode_indexed(self, hidden_states):
        """Decode routing for the indexed experts.

        hidden_states: [1, 1, 1 (32), hidden] L1 interleaved (one token).
        Returns ([1, k] UINT16 row-major expert ids, [1, 1, 1, k] row-major BF16 softmax weights in DRAM)."""
        n_tiles = self.decode_width // ttnn.TILE_SIZE
        logits = ttnn.linear(
            hidden_states,
            self.weight_decode,
            bias=self.bias_decode,
            program_config=matmul_1d_program_config(None, self.hidden_dim, 45, grid=(n_tiles, 1, 1)),
            # The router keeps the default (HiFi2) fidelity of the unfused graph.
            compute_kernel_config=auto_matmul_compute_config(hidden_states.device(), operands_low_precision=False),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        logits = ttnn.reshape(logits, (self.decode_tokens, 16, 16))
        logits_sharded = ttnn.to_memory_config(logits, self.gate_memory_config)
        logits.deallocate(True)
        scores, indices = ttnn.experimental.deepseek.moe.generalized_moe_gate(
            logits_sharded,
            bias_tensor=self.gate_selection_bias,
            input_indices_tensor=self.gate_expert_ids,
            output_tensor=self.gate_scores,
            output_indices_tensor=self.gate_indices,
            eps=1e-20,
            scaling_factor=1.0,
            enable_sigmoid=False,
            topk=self.top_k,
            output_softmax=True,
        )
        logits_sharded.deallocate(True)
        # The top-k ids / scores are the first k entries of row 0 of each token's buffer.
        k, b = self.top_k, self.decode_tokens
        indices_rm = ttnn.slice(indices, [0, 0, 0], [b, 1, k], memory_config=ttnn.L1_MEMORY_CONFIG)
        scores_rm = ttnn.slice(scores, [0, 0, 0], [b, 1, k], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(indices_rm, (b, k)), ttnn.reshape(scores_rm, (b, 1, 1, k))

    def _fused_call(self, hidden_states, use_throughput_experts):
        """Forward pass using fused matmul+topk+softmax kernel.

        Note: Fused op only supports throughput experts (sparse [B,k] output).
        """
        if not use_throughput_experts:
            raise ValueError("Fused topk_router_gpt requires use_throughput_experts=True")

        # Typecast to bf16 if needed (fused op requires bf16 input)
        needs_typecast = hidden_states.dtype != ttnn.bfloat16
        if needs_typecast:
            hidden_states_bf16 = ttnn.typecast(hidden_states, dtype=ttnn.bfloat16)
        else:
            hidden_states_bf16 = hidden_states

        B = hidden_states_bf16.shape[0]
        device = hidden_states_bf16.device()

        # Initialize fused op buffers (once)
        self._init_fused_op(device, B)

        # Run fused matmul + topk + softmax
        indices_rm, weights_rm = ttnn.experimental.topk_router_gpt(
            hidden_states_bf16,
            weight_tensor=self.weight,
            bias_tensor=self._fused_bias,
            k=self.top_k,
            num_experts=self.num_experts,
        )

        if needs_typecast:
            ttnn.deallocate(hidden_states_bf16)

        # Kernel produces uint16 RM [B, k_padded] and bf16 RM [B, k_padded].
        # Slice to [B, top_k] in RM and return directly.
        # fused_decode.py handles RM input natively (zero-cost reshape to 4D).
        expert_indices = ttnn.slice(indices_rm, [0, 0], [B, self.top_k])
        expert_weights = ttnn.slice(weights_rm, [0, 0], [B, self.top_k])
        ttnn.deallocate(indices_rm)
        ttnn.deallocate(weights_rm)
        return expert_indices, expert_weights
