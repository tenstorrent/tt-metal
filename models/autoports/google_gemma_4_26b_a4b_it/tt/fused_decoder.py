# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Graph-fused single-device decoder for ``google/gemma-4-26B-A4B-it``.

This stage deliberately inherits the functional decoder's public tensor,
paged-cache, non-aligned-length, and trace contracts.  The runtime override
removes the standalone dense-MLP GELU dispatch by applying Gemma4's
``gelu_pytorch_tanh`` unary on the first operand inside the consuming multiply.
No functional-decoder forward method is used as a fallback: ``from_state_dict``
constructs this class and normal virtual dispatch reaches the fused primitive
from both prefill and decode.
"""

from __future__ import annotations

import ttnn

from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    HIDDEN_SIZE,
    MOE_INTERMEDIATE_SIZE,
    NUM_EXPERTS,
    TILE_SIZE,
    TOP_K_EXPERTS,
    FunctionalDecoder,
    _build_sparse_matmul_config,
)


class FusedDecoder(FunctionalDecoder):
    """Functional-equivalent decoder with verified single-device graph fusions."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Fold immutable router scales once at setup, removing a broadcast
        # multiply from every prefill/decode invocation.
        self.fused_router_scale = ttnn.mul(
            self.weights.router_scale,
            self.router_hidden_scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _router_weights(self, residual: ttnn.Tensor) -> ttnn.Tensor:
        tokens = residual.shape[-2]
        router_in = self._rms_norm(residual, None)
        router_in = ttnn.mul(router_in, self.fused_router_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        router_in = ttnn.reshape(router_in, [tokens, HIDDEN_SIZE])
        router_in = ttnn.typecast(router_in, ttnn.float32)
        logits = ttnn.linear(
            router_in,
            self.weights.router_proj,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logits = ttnn.typecast(logits, ttnn.bfloat16)
        top_values, top_indices = ttnn.topk(logits, k=TOP_K_EXPERTS, dim=-1, sorted=True)
        top_values = ttnn.softmax(
            top_values,
            dim=-1,
            numeric_stable=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=top_indices, src=top_values)
        routing = ttnn.mul(routing, self.weights.router_per_expert_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        routing = ttnn.typecast(routing, ttnn.bfloat16)
        return ttnn.reshape(routing, [1, 1, tokens, NUM_EXPERTS])

    def _dense_mlp(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.linear(
            x,
            self.weights.mlp_gate,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            x,
            self.weights.mlp_up,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0)],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.linear(
            hidden,
            self.weights.mlp_down,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _moe_decode_single_user(
        self,
        hidden_states: ttnn.Tensor,
        routing_weights: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Sparse MoE decode with GeGLU folded into the consuming multiply."""
        batch = hidden_states.shape[2]
        sparsity = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        gate_up_config = _build_sparse_matmul_config(batch, MOE_INTERMEDIATE_SIZE)
        down_config = _build_sparse_matmul_config(batch, HIDDEN_SIZE)

        gate = ttnn.sparse_matmul(
            hidden_states,
            self.weights.expert_gate,
            sparsity=sparsity,
            nnz=TOP_K_EXPERTS,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_config,
            dtype=self.activation_dtype,
            compute_kernel_config=self.correctness_compute_config,
        )
        sparse_intermediate = gate.shape[-1]
        gate = ttnn.reshape(gate, (batch, NUM_EXPERTS, 1, sparse_intermediate))
        gate = ttnn.transpose(gate, 1, 2)
        gate = ttnn.reshape(gate, (batch, NUM_EXPERTS, sparse_intermediate))

        up = ttnn.sparse_matmul(
            hidden_states,
            self.weights.expert_up,
            sparsity=sparsity,
            nnz=TOP_K_EXPERTS,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_config,
            dtype=self.activation_dtype,
        )
        up = ttnn.reshape(up, (batch, NUM_EXPERTS, 1, sparse_intermediate))
        up = ttnn.transpose(up, 1, 2)
        up = ttnn.reshape(up, (batch, NUM_EXPERTS, sparse_intermediate))
        down_input = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0)],
        )
        down_input = ttnn.transpose(down_input, 1, 0)
        down_input = ttnn.reshape(down_input, (1, NUM_EXPERTS, batch, sparse_intermediate))

        down = ttnn.sparse_matmul(
            down_input,
            self.weights.expert_down,
            sparsity=sparsity,
            nnz=TOP_K_EXPERTS,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=down_config,
            is_input_a_sparse=True,
            dtype=self.activation_dtype,
        )
        next_states = ttnn.permute(down, (0, 2, 1, 3))
        next_states = ttnn.reshape(next_states, (batch, NUM_EXPERTS, HIDDEN_SIZE))
        routing_3d = ttnn.reshape(routing_weights, (batch, NUM_EXPERTS, 1))
        next_states = ttnn.mul(next_states, routing_3d)
        next_states = ttnn.sum(next_states, dim=1)
        next_states = ttnn.unsqueeze_to_4D(next_states)
        return ttnn.reshape(
            next_states,
            (1, 1, batch, HIDDEN_SIZE),
            (1, 1, max(TILE_SIZE, batch), HIDDEN_SIZE),
        )

    def _moe_prefill_chunk(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        """All-expert sparse prefill with GeGLU folded into its multiply."""
        if hidden_states.shape[2] > TILE_SIZE:
            hidden_chunks = ttnn.split(hidden_states, TILE_SIZE, dim=2)
            routing_chunks = ttnn.split(routing_weights, TILE_SIZE, dim=2)
        else:
            hidden_chunks = [hidden_states]
            routing_chunks = [routing_weights]

        result = None
        for hidden_chunk, routing_chunk in zip(hidden_chunks, routing_chunks):
            chunk_result = self._moe_prefill_tile_group(hidden_chunk, routing_chunk)
            if result is None:
                result = chunk_result
            else:
                concatenated = ttnn.concat([result, chunk_result], dim=2)
                result.deallocate(True)
                chunk_result.deallocate(True)
                result = concatenated
        return result

    def _moe_prefill_tile_group(
        self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor
    ) -> ttnn.Tensor:
        chunk_len = hidden_states.shape[2]
        group_size = chunk_len // TILE_SIZE
        hidden_grouped = ttnn.reshape(hidden_states, (1, group_size, TILE_SIZE, HIDDEN_SIZE))
        sparsity = ttnn.repeat(self.expert_prefill_sparsity, (1, 1, group_size, 1))
        nnz = NUM_EXPERTS * group_size
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        gate_up_config = _build_sparse_matmul_config(TILE_SIZE, MOE_INTERMEDIATE_SIZE)
        down_config = _build_sparse_matmul_config(TILE_SIZE, HIDDEN_SIZE)

        gate = ttnn.sparse_matmul(
            hidden_grouped,
            self.weights.expert_gate,
            sparsity=sparsity,
            nnz=nnz,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_config,
            dtype=ttnn.bfloat16,
        )
        sparse_intermediate = gate.shape[-1]
        gate = ttnn.transpose(gate, 1, 3)
        gate = ttnn.reshape(gate, (1, NUM_EXPERTS, chunk_len, sparse_intermediate))

        up = ttnn.sparse_matmul(
            hidden_grouped,
            self.weights.expert_up,
            sparsity=sparsity,
            nnz=nnz,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_config,
            dtype=ttnn.bfloat16,
        )
        hidden_grouped.deallocate(True)
        up = ttnn.transpose(up, 1, 3)
        up = ttnn.reshape(up, (1, NUM_EXPERTS, chunk_len, sparse_intermediate))
        down_input = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0)],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        down_input = ttnn.to_layout(down_input, ttnn.TILE_LAYOUT)
        down_input = ttnn.reshape(down_input, (1, NUM_EXPERTS, chunk_len, sparse_intermediate))

        down = ttnn.sparse_matmul(
            down_input,
            self.weights.expert_down,
            sparsity=self.expert_prefill_sparsity,
            nnz=NUM_EXPERTS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=down_config,
            is_input_a_sparse=True,
            dtype=ttnn.bfloat16,
        )
        down_input.deallocate(True)
        next_states = ttnn.reshape(down, (1, NUM_EXPERTS, chunk_len, HIDDEN_SIZE))
        routing_permuted = ttnn.permute(routing_weights, (0, 3, 2, 1))
        next_states = ttnn.mul(next_states, routing_permuted)
        next_states = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(next_states, dims=[1]))
        return ttnn.reshape(next_states, (1, 1, chunk_len, HIDDEN_SIZE))


__all__ = ["FusedDecoder"]
