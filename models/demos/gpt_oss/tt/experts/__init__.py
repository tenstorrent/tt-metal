# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generic MoE Experts Module.

This module provides a model-agnostic implementation of MoE experts.
Models provide their own ProgramConfig implementations for customization.

Usage:
    from models.demos.gpt_oss.tt.experts import Experts, ExpertConfig
    from models.demos.gpt_oss.tt.expert_configs import GPTOSSProgramConfig

    config = ExpertConfig(...)
    program_config = GPTOSSProgramConfig()

    experts = Experts(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=program_config,
    )

    output = experts(hidden_states, routing_weights)
"""

import ttnn

from models.demos.gpt_oss.config import MeshConfig, ModeConfig

from .config import ExpertConfig, ProgramConfig
from .decode import DenseDecodePlan, decode_forward, decode_forward_dense, decode_forward_stream
from .operations import apply_tensor_parallel_allreduce
from .prefill import prefill_forward
from .weights import load_decode_expert_weights, load_dense_decode_expert_weights, load_expert_weights

__all__ = ["Experts", "ExpertConfig", "ProgramConfig"]


class Experts:
    """
    Generic MoE Expert implementation with automatic decode/prefill dispatch.

    This class provides a clean interface for expert layers. Models provide
    their own ProgramConfig implementations to customize behavior.
    """

    def __init__(
        self,
        mesh_device,
        config: ExpertConfig,
        state_dict,
        ccl_manager,
        mesh_config: MeshConfig,
        program_config: ProgramConfig,
        weight_dtype=ttnn.bfloat4_b,
        tensor_cache_path=None,
        indexed_decode=False,
        dense_decode_tokens=None,
    ):
        """
        Initialize expert layers.

        Args:
            mesh_device: TTNN mesh device
            config: Expert configuration
            state_dict: Expert weights dictionary
            ccl_manager: Communication manager
            mesh_config: Mesh parallelization configuration
            program_config: Model-specific program configurations
            weight_dtype: Data type for weights (default: bfloat4_b)
            tensor_cache_path: Optional path for weight caching
            indexed_decode: Fused decode: only the routed experts are computed (DRAM-streaming ops, one token per
                device, TP without expert parallelism); requires the router's decode_indexed outputs.
            dense_decode_tokens: Batched decode with this many tokens per device (fused_decode.config
                dense_moe_decode_supported): every expert is computed for every token by two DRAM-sharded matmuls
                over all experts side by side (decode_forward_dense).
        """
        self.config = config
        self.mesh_config = mesh_config
        self.weight_dtype, self.tensor_cache_path = weight_dtype, tensor_cache_path
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.program_config = program_config

        # Load weights. The batched dense path (dense_decode_tokens) serves prefill too, from its own layout, so the
        # per-expert weights are not loaded there (they are ~2.5 GB per device for gpt-oss-20b at TP=4, the margin a
        # 128k-token context needs at batch 32).
        self.weights = None
        self.prefill_sparsity = None
        if not dense_decode_tokens:
            self.weights = load_expert_weights(
                mesh_device=mesh_device,
                config=config,
                state_dict=state_dict,
                mesh_config=mesh_config,
                weight_dtype=weight_dtype,
                tensor_cache_path=tensor_cache_path,
            )
            # Cache prefill sparsity (created once, reused for all prefill calls)
            self.prefill_sparsity = self._create_prefill_sparsity()

        # Fused decode (TP over mesh columns, no expert parallelism): only the routed experts are computed, by the
        # DRAM-streaming ops of fused_decode/stream.py (ops and buffers shared by every layer through the CCL manager).
        self.decode_weights = None
        self.decode_stream = None
        if indexed_decode:
            assert mesh_config.decode.tp > 1 and mesh_config.decode.ep == 1, "fused decode needs TP > 1, EP == 1"
            self.decode_weights = load_decode_expert_weights(
                mesh_device=mesh_device,
                config=config,
                state_dict=state_dict,
                mesh_config=mesh_config,
                weight_dtype=weight_dtype,
                tensor_cache_path=tensor_cache_path,
            )
            self.decode_stream = ccl_manager.get_decode_expert_stream(
                config.hidden_size,
                self.decode_weights.intermediate_padded,
                config.num_experts_per_tok,
                config.swiglu_limit,
                config.alpha,
            )

        self.dense_decode_weights = None
        if dense_decode_tokens:
            self.dense_decode_weights = load_dense_decode_expert_weights(
                mesh_device=mesh_device,
                config=config,
                state_dict=state_dict,
                mesh_config=mesh_config,
                tokens=dense_decode_tokens,
                weight_dtype=weight_dtype,
                tensor_cache_path=tensor_cache_path,
            )
            self.dense_decode_plan = DenseDecodePlan(
                mesh_device,
                config.hidden_size,
                self.dense_decode_weights.intermediate_local,
                config.num_experts,
                dense_decode_tokens,
            )

        # For backward compatibility
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size

    def _create_prefill_sparsity(self):
        """Create prefill sparsity mask once and reuse."""
        import torch
        from models.demos.gpt_oss.config import Mode

        prefill_config = self.mesh_config.get_config(Mode.PREFILL)
        prefill_ep = prefill_config.ep
        tokens_per_ep = self.config.num_experts // prefill_ep

        sparsity = torch.zeros(1, 1, prefill_ep, self.config.num_experts)
        for i in range(prefill_ep):
            sparsity[:, :, i, i * tokens_per_ep : (i + 1) * tokens_per_ep] = torch.ones(1, 1, 1, tokens_per_ep)

        return ttnn.from_torch(
            sparsity,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                dims=(-2, None) if prefill_ep > 1 else (None, None),
                mesh_shape=self.mesh_device.shape,
                mesh_device=self.mesh_device,
            ),
        )

    def __call__(
        self,
        hidden_states,
        topk_expert_weights: ttnn.Tensor,
        is_decode: bool = True,
        topk_expert_indices: ttnn.Tensor = None,
    ):
        """
        Forward pass - automatically dispatches to decode or prefill.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            topk_expert_weights: Sparse router scores output [seq_len, num_experts]
            is_decode: Decode mode
            topk_expert_indices: Top-k expert indices per token (unused in this version of experts)

        Returns:
            Expert output tensor [1, batch, seq_len, hidden_size]
        """
        if is_decode and self.decode_stream is not None:
            return decode_forward_stream(
                hidden_states,
                topk_expert_indices,
                topk_expert_weights,
                weights=self.decode_weights,
                stream=self.decode_stream,
                send=self.ccl_manager.decode_boundary_send(self.config.hidden_size, "moe"),
            )
        if is_decode and self.dense_decode_weights is not None:
            partial = decode_forward_dense(
                hidden_states, topk_expert_weights, self.dense_decode_weights, self.dense_decode_plan, self.config
            )
            if getattr(self, "decode_all_reduce", None) is not None:
                # Batched decode layer (batched_decode.py): one-op all-reduce into the sharded residual layout.
                return self.decode_all_reduce(partial, "moe")
            out = apply_tensor_parallel_allreduce(
                partial, self.mesh_config, self.mesh_device, partial.shape[-2], self.ccl_manager
            )
            tokens = out.shape[-2]
            return ttnn.reshape(
                out, (1, 1, tokens, self.config.hidden_size), (1, 1, max(32, tokens), self.config.hidden_size)
            )
        if self.dense_decode_weights is not None:
            return self._prefill_forward_dense(hidden_states, topk_expert_weights)
        # Determine mode based on sequence length. Decode with several tokens per device (batch > 1 without expert
        # parallelism) takes the prefill path: each token's experts are the same computation either way, and the
        # one-token decode ops below only accept a single token.
        if is_decode and hidden_states.shape[2] == 1:
            return decode_forward(
                hidden_states=hidden_states,
                routing_weights=topk_expert_weights,
                weights=self.weights,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
                program_config=self.program_config,
            )
        else:
            return prefill_forward(
                hidden_states=hidden_states,
                routing_weights=topk_expert_weights,
                weights=self.weights,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
                program_config=self.program_config,
                prefill_sparsity=self.prefill_sparsity,
            )

    def _prefill_forward_dense(self, hidden_states, routing_weights):
        """Prefill MoE of the batched dense path: the sequence in tiles of plan tokens, each through
        decode_forward_dense (every expert, from the side-by-side weights), then one TP all-reduce.

        hidden_states: [1, 1, S, hidden] with S a multiple of the plan's tokens; routing_weights: [S, num_experts]."""
        plan, hidden = self.dense_decode_plan, self.config.hidden_size
        tokens = plan.tokens
        seq_len = hidden_states.shape[-2]
        assert seq_len % tokens == 0, f"prefill length {seq_len} is not a multiple of {tokens}"
        experts = routing_weights.shape[-1]
        routing_weights = ttnn.reshape(routing_weights, (seq_len, experts))
        parts, groups = [], []
        for start in range(0, seq_len, tokens):
            x = ttnn.slice(hidden_states, [0, 0, start, 0], [1, 1, start + tokens, hidden])
            w = ttnn.slice(routing_weights, [start, 0], [start + tokens, experts])
            parts.append(decode_forward_dense(x, w, self.dense_decode_weights, plan, self.config))
            x.deallocate(True)
            if len(parts) == 64:
                groups.append(ttnn.concat(parts, dim=2))
                for t in parts:
                    t.deallocate(True)
                parts = []
        if parts:
            groups.append(ttnn.concat(parts, dim=2) if len(parts) > 1 else parts[0])
            if len(parts) > 1:
                for t in parts:
                    t.deallocate(True)
        partial = groups[0] if len(groups) == 1 else ttnn.concat(groups, dim=2)
        if len(groups) > 1:
            for t in groups:
                t.deallocate(True)
        out = apply_tensor_parallel_allreduce(partial, self.mesh_config, self.mesh_device, seq_len, self.ccl_manager)
        return ttnn.reshape(out, (1, 1, seq_len, hidden), (1, 1, max(32, seq_len), hidden))
