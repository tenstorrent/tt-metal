# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode forward pass for experts (seq_len=1)."""

import ttnn
from models.demos.gpt_oss.config import Mode

from .config import ExpertConfig, ProgramConfig
from .operations import apply_expert_parallel_allreduce, apply_swiglu, apply_tensor_parallel_allreduce
from .weights import ExpertWeights


def decode_forward(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config: ExpertConfig,
    mesh_config,
    mesh_device,
    ccl_manager,
    program_config: ProgramConfig,
):
    """
    Decode forward pass - optimized for single token (seq_len=1).

    Args:
        hidden_states: Input tensor [batch, 1, hidden_size]
        routing_weights: Router output [batch, num_experts]
        weights: Expert weights
        config: Expert configuration
        mesh_config: Mesh parallelization config
        mesh_device: TTNN mesh device
        ccl_manager: Communication manager
        program_config: Model-specific program configs

    Returns:
        Expert output [1, batch, 1, hidden_size]
    """
    activation_dtype = ttnn.bfloat8_b
    batch_dim = 1
    seq_dim = 2
    batch_size = hidden_states.shape[batch_dim]
    seq_len = hidden_states.shape[seq_dim]

    if seq_len != 1:
        raise ValueError(f"Decode mode requires seq_len=1, got {seq_len}")

    # Get parallelization config
    mode_config = mesh_config.get_config(Mode.DECODE)
    ep, tp = mode_config.ep, mode_config.tp
    # Prepare inputs for sparse matmul
    sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing_weights), ttnn.ROW_MAJOR_LAYOUT)

    # EP-specific routing remap for sparsity
    if ep > 1:
        sparsity = ttnn.moe_routing_remap(ttnn.reshape(sparsity, (1, sparsity.shape[-1])), 4, 4, 0)
        routing_weights = ttnn.tilize_with_zero_padding(sparsity, use_multicore=True)

    # nnz is the total non-zero entries in the sparsity tensor across the full
    # batched matmul. With B tokens routed to top-K experts each, that's K*B
    # active (token, expert) outputs (clamped at K*B / ep when expert-parallel).
    num_experts_per_tok = (config.num_experts_per_tok * batch_size) // ep
    output_tile = ttnn.Tile([32, 32])

    # Gate/up matmul outputs scale as B * num_experts * 32 * intermediate. For
    # B=1 they fit in L1 comfortably (~3 MB on 20b). At B=32 they're ~94 MB
    # each — way past L1. Switch to DRAM once batch size makes L1 untenable.
    matmul_mem_config = ttnn.L1_MEMORY_CONFIG if batch_size <= 4 else ttnn.DRAM_MEMORY_CONFIG

    # Gate projection
    gate = ttnn.sparse_matmul(
        hidden_states,
        weights.gate_proj,
        sparsity=sparsity,
        nnz=num_experts_per_tok,
        memory_config=matmul_mem_config,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(
            hidden_states.shape[2], weights.gate_proj.shape[3], k=hidden_states.shape[-1]
        ),
        dtype=activation_dtype,
    )
    # sparse_matmul on hidden_states=[1, B, 1, hidden] @ weights=[1, num_experts, hidden, inter]
    # produces a rank-6 output [1, B, 1, num_experts, 1, inter]. Drop the two
    # leading singleton batch dims (added by the kernel from the leading 1s on
    # both inputs) via squeeze — reshape rejects this for B>1 even though the
    # logical volumes match.
    gate = ttnn.squeeze(gate, 0)  # → [B, 1, num_experts, 1, inter]
    gate = ttnn.squeeze(gate, 1)  # → [B, num_experts, 1, inter]
    gate = ttnn.transpose(gate, 1, 2)  # → [B, 1, num_experts, inter]
    gate = ttnn.squeeze(gate, 1)  # → [B, num_experts, inter]
    gate = ttnn.add(gate, weights.gate_proj_bias, output_tensor=gate)

    # Up projection
    up = ttnn.sparse_matmul(
        hidden_states,
        weights.up_proj,
        sparsity=sparsity,
        nnz=num_experts_per_tok,
        memory_config=matmul_mem_config,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(
            hidden_states.shape[2], weights.up_proj.shape[3], k=hidden_states.shape[-1]
        ),
        dtype=activation_dtype,
    )
    hidden_states.deallocate(True)
    # Same rank-6 → rank-4 squeeze as gate above.
    up = ttnn.squeeze(up, 0)
    up = ttnn.squeeze(up, 1)
    up = ttnn.transpose(up, 1, 2)
    up = ttnn.squeeze(up, 1)
    up = ttnn.add(up, weights.up_proj_bias, output_tensor=up)

    # Apply SwiGLU activation (consumes gate and up internally)
    down_input = apply_swiglu(gate, up, config)
    # down_input is [B, num_experts, inter]. The down sparse matmul uses
    # is_input_a_sparse=True, where the sparsity tensor maps over A's batch
    # dims (all dims except the last 2). To match our [B, num_experts]
    # sparsity, A must be shaped [B, num_experts, M, inter] with M=1, so
    # batch_length_A = B*num_experts == sparsity volume.
    down_input = ttnn.reshape(
        down_input,
        (batch_size, config.num_experts, seq_len, weights.intermediate_size_per_device),
    )
    # Down projection. is_input_a_sparse=True with is_input_b_sparse=False (the
    # default flips the latter to True) makes the kernel use A's batch dims
    # for the sparsity check — for B>1 we need the sparsity to span
    # [B, num_experts] = batch_length_A, not just num_experts.
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        nnz=num_experts_per_tok,
        memory_config=matmul_mem_config,
        output_tile=output_tile,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
        program_config=program_config.get_decode_down_config(
            down_input.shape[2], weights.down_proj.shape[-1], k=down_input.shape[-1]
        ),
        dtype=activation_dtype,
    )

    down_input.deallocate(True)
    sparsity.deallocate(True)
    # down output shape is [B, num_experts, 1, hidden] (from the sparse-A
    # rank-4 batched matmul above). Drop the M=1 dim to get [B, num_experts, hidden].
    next_states = ttnn.squeeze(down, 2)
    next_states = ttnn.add(next_states, weights.down_proj_bias, output_tensor=next_states)
    # routing_weights enters here as [B, num_experts]. The previous permute(1,0)
    # was a no-op for B=1 but reorders elements for B>1. We need
    # [B, num_experts, 1] preserving the (B, num_experts) layout, which is just
    # an unsqueeze on the last dim.
    routing_weights = ttnn.reshape(routing_weights, (batch_size, config.num_experts, 1))

    next_states = ttnn.mul(next_states, routing_weights, output_tensor=next_states)
    routing_weights.deallocate(True)

    # Reduce across experts
    next_states = ttnn.sum(next_states, dim=1)
    # Note: unsqueeze_to_4D typically returns a view, so we don't deallocate the sum result
    next_states = ttnn.unsqueeze_to_4D(next_states)

    # Expert parallel communication
    if ep > 1:
        next_states = apply_expert_parallel_allreduce(next_states, mesh_config, ccl_manager)

    # Note: unsqueeze_to_4D typically returns a view
    next_states = ttnn.unsqueeze_to_4D(next_states)

    # Tensor parallel communication
    if tp > 1:
        # Note: apply_tensor_parallel_allreduce already handles deallocating the input tensor
        next_states = apply_tensor_parallel_allreduce(
            next_states,
            mesh_config,
            mesh_device,
            seq_len,
            ccl_manager,
        )

    # Final reshape
    # Note: reshape typically returns a view, so we don't deallocate the original
    next_states = ttnn.reshape(
        next_states,
        (1, batch_size, seq_len, config.hidden_size),
        (1, batch_size, max(32, seq_len), config.hidden_size),
    )

    return next_states
