# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode forward pass for experts (seq_len=1)."""

import ttnn
from models.demos.gpt_oss.config import Mode

from .config import ExpertConfig, ProgramConfig
from .operations import apply_expert_parallel_allreduce, apply_tensor_parallel_allreduce
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
    topk_expert_indices=None,
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

    # ✅ Use exceptions instead of assertions
    if seq_len != 1:
        raise ValueError(f"Decode mode requires seq_len=1, got {seq_len}")
    if batch_size != 1:
        raise NotImplementedError(f"Currently only batch_size=1 supported, got {batch_size}")

    # Get parallelization config
    mode_config = mesh_config.get_config(Mode.DECODE)
    ep, tp = mode_config.ep, mode_config.tp
    # Prepare inputs for sparse matmul
    # hidden_states_4D = ttnn.unsqueeze_to_4D(hidden_states)
    sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing_weights), ttnn.ROW_MAJOR_LAYOUT)

    # EP-specific routing remap for sparsity
    if ep > 1:
        sparsity = ttnn.moe_routing_remap(ttnn.reshape(sparsity, (1, sparsity.shape[-1])), 4, 4, 0)
        routing_weights = ttnn.tilize_with_zero_padding(sparsity, use_multicore=True)

    num_experts_per_tok = config.num_experts_per_tok // ep
    output_tile = ttnn.Tile([32, 32])

    # Gate projection
    # Fused gate+up: one sparse_matmul over the concatenated [gate|up] weight.
    # gate/up output kept bf16 (not bf8): the downstream transpose+slice+add+SwiGLU
    # chain operates in bf16 and otherwise inserts bf8->bf16 typecasts around each
    # transpose (tracy: 144 Transpose->Typecast->Transpose in decode = ~1.2ms/tok).
    # Emitting bf16 directly from the matmul removes those casts. Verify perf+accuracy.
    gate = ttnn.sparse_matmul(
        hidden_states,
        weights.gate_up_proj,
        sparsity=sparsity,
        # Static nnz = num_experts_per_tok. The historical reason for nnz=None was a
        # deadlock when the actual non-zero sparsity count < nnz (the in0-mcast
        # receivers loop nnz times while the sender only mcasts for real non-zeros;
        # see #45943/#45052). MEASURED on this model: a probe over 1704 decode
        # sparse_matmul calls found the non-zero count is NEVER below 4
        # (distribution {4: 1392, 32: 120, 128: 96, 1024: 48, 2048: 48}, min=4), so
        # nnz=4 cannot under-run. Inferring it instead costs a device-side reduction
        # + FILL per call: SparseMatmul 6.159 -> 5.827 ms/tok, decode 15.854 ->
        # 15.518, 58.9 -> 60.0 tok/s/user, accuracy unchanged (0.9667 / 1.0000).
        nnz=num_experts_per_tok,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(
            hidden_states.shape[2], weights.gate_up_proj.shape[3], k=hidden_states.shape[-1]
        ),
        dtype=ttnn.bfloat16,
    )
    hidden_states.deallocate(True)
    # Fused output is [.., 2*I]; split into gate/up along the last dim after the
    # shared reshape/transpose (halves the gate/up sparse_matmul launches).
    I = weights.intermediate_size_per_device
    # Fused expert-tail SwiGLU (custom generic_op): transpose-eliminating +
    # expert-skipping + scatter. Reads gate/up directly from the raw fused matmul
    # output [1,E,1,2I] (native layout, no transpose/slice), processes ONLY the
    # active experts (ids from the router top-k device tensor), and scatters
    # silu(clamp(gate,max=a*lim))*(clamp(up,-lim,lim)+1) to the active expert slots
    # of down_input [1,E,1,I]. Bias (concat[gate|up]) added once via the prebuilt
    # weights.gateup_bias. Replaces ~10 ops/layer (transpose+slice x2+2 bias+clamp x2
    # +silu+ (+1)+mul) with a single generic_op over the active experts.
    from models.demos.gpt_oss.kernels.swiglu_final_op import fused_swiglu_final

    raw = ttnn.reshape(gate, (batch_size, config.num_experts, 1, 2 * I))
    # Prep active-expert id list -> [1,1,1,nact] uint32 ROW_MAJOR device tensor.
    _nact = config.num_experts_per_tok
    _idx = ttnn.to_layout(topk_expert_indices, ttnn.ROW_MAJOR_LAYOUT)
    _idx = ttnn.typecast(_idx, ttnn.uint32)
    _idx = ttnn.reshape(_idx, (1, 1, 1, _nact))
    down_input = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, config.num_experts, seq_len, I]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        mesh_device,
        ttnn.L1_MEMORY_CONFIG,
    )
    _cap = config.alpha * config.swiglu_limit
    # Bias folded inside the kernel (weights.gateup_bias), only for active experts.
    fused_swiglu_final(raw, weights.gateup_bias, _idx, down_input, _nact, _cap, config.swiglu_limit)
    raw.deallocate(True)
    # Down projection
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        # Static nnz: see the gate_up call above. The measured non-zero count never
        # falls below num_experts_per_tok on this model, so the #45943/#45052
        # under-run deadlock cannot trigger. Verified with 6 full demo runs + the
        # accuracy test, no hang.
        nnz=num_experts_per_tok,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        is_input_a_sparse=True,
        program_config=program_config.get_decode_down_config(
            down_input.shape[2], weights.down_proj.shape[-1], k=down_input.shape[-1]
        ),
        # bf16 output (not bf8): feeds the bf16 permute+add(bias)+mul(routing)+sum tail;
        # emitting bf16 avoids the per-op bf8->bf16 recast (same win as gate/up above).
        dtype=ttnn.bfloat16,
    )

    down_input.deallocate(True)
    sparsity.deallocate(True)
    # Apply bias and routing weights
    # Note: permute/reshape operations return views - do not deallocate originals
    next_states = ttnn.permute(down, (0, 2, 1, 3))
    next_states = ttnn.reshape(next_states, (batch_size, config.num_experts, config.hidden_size))
    next_states = ttnn.add(next_states, weights.down_proj_bias, output_tensor=next_states)
    routing_weights = ttnn.permute(routing_weights, (1, 0))
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
