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

    # ✅ Use exceptions instead of assertions
    if seq_len != 1:
        raise ValueError(f"Decode mode requires seq_len=1, got {seq_len}")

    # Gate/up outputs scale as B * num_experts * 32 * intermediate_per_device. At B=1
    # they fit in L1 comfortably; by B=32 they do not. Spill to DRAM once the batch
    # makes L1 untenable, keeping the existing B=1 path on L1 unchanged.
    matmul_mem_config = ttnn.L1_MEMORY_CONFIG if batch_size <= 4 else ttnn.DRAM_MEMORY_CONFIG

    # Get parallelization config
    mode_config = mesh_config.get_config(Mode.DECODE)
    ep, tp = mode_config.ep, mode_config.tp
    # Prepare inputs for sparse matmul
    # hidden_states_4D = ttnn.unsqueeze_to_4D(hidden_states)
    sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing_weights), ttnn.ROW_MAJOR_LAYOUT)

    # EP-specific routing remap for sparsity
    if ep > 1:
        if batch_size > 1:
            # moe_routing_remap load-balances a token's non-zero experts across the EP
            # devices, and its data-dependent split is defined for a single token:
            # moe_routing_remap_device_operation.cpp:21 asserts the input is [1, E].
            # Batched decode on an expert-parallel mesh therefore needs that op extended
            # to a leading batch dim (one balanced split per row). Until then, multi-user
            # decode on an EP mesh must use the throughput experts path.
            raise NotImplementedError(
                f"Batched decode (batch={batch_size}) with expert parallelism (ep={ep}) requires a batched "
                "moe_routing_remap; the op currently accepts only [1, num_experts]. Use throughput experts "
                "on this mesh, or run with ep=1."
            )
        sparsity = ttnn.moe_routing_remap(
            ttnn.reshape(sparsity, (1, sparsity.shape[-1])),
            config.num_experts_per_tok,
            ep,
            mesh_config.ep_axis,
        )
        routing_weights = ttnn.tilize_with_zero_padding(sparsity, use_multicore=True)

    num_experts_per_tok = config.num_experts_per_tok // ep
    output_tile = ttnn.Tile([32, 32])

    # Gate projection
    gate = ttnn.sparse_matmul(
        hidden_states,
        weights.gate_proj,
        sparsity=sparsity,
        # nnz intentionally omitted (None -> inferred at runtime). Passing a static
        # nnz makes the sparse_matmul in0-mcast receivers loop a fixed count while the
        # sender only mcasts for the *actual* non-zero `sparsity` entries. The decode
        # routing weights (softmax over top-k, scattered) frequently have <k non-zeros
        # on Blackhole (small weights flush to 0), so a static nnz != actual count and
        # the receivers deadlock in noc_semaphore_wait. Inferring the count is robust.
        # See tenstorrent/tt-metal#45943 (op deadlock) / #45052 (gpt-oss hang).
        nnz=None,
        memory_config=matmul_mem_config,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(
            hidden_states.shape[2], weights.gate_proj.shape[3], k=hidden_states.shape[-1]
        ),
        dtype=activation_dtype,
    )
    # sparse_matmul on hidden=[1, B, 1, H] @ weights=[1, num_experts, H, I] produces a
    # rank-6 output [1, B, 1, num_experts, 1, I]. Drop the two leading singleton dims the
    # kernel adds from the leading 1s on both inputs; ttnn.reshape rejects this multi-dim
    # collapse for B>1 even though the logical volumes match (it cannot view across
    # reordered batch dims in TILE layout).
    gate = ttnn.squeeze(gate, 0)  # -> [B, 1, num_experts, 1, I]
    gate = ttnn.squeeze(gate, 1)  # -> [B, num_experts, 1, I]
    gate = ttnn.transpose(gate, 1, 2)  # -> [B, 1, num_experts, I]
    gate = ttnn.squeeze(gate, 1)  # -> [B, num_experts, I]
    gate = ttnn.add(gate, weights.gate_proj_bias, output_tensor=gate)

    # Up projection
    up = ttnn.sparse_matmul(
        hidden_states,
        weights.up_proj,
        sparsity=sparsity,
        # nnz intentionally omitted (None -> inferred at runtime). Passing a static
        # nnz makes the sparse_matmul in0-mcast receivers loop a fixed count while the
        # sender only mcasts for the *actual* non-zero `sparsity` entries. The decode
        # routing weights (softmax over top-k, scattered) frequently have <k non-zeros
        # on Blackhole (small weights flush to 0), so a static nnz != actual count and
        # the receivers deadlock in noc_semaphore_wait. Inferring the count is robust.
        # See tenstorrent/tt-metal#45943 (op deadlock) / #45052 (gpt-oss hang).
        nnz=None,
        memory_config=matmul_mem_config,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(
            hidden_states.shape[2], weights.up_proj.shape[3], k=hidden_states.shape[-1]
        ),
        dtype=activation_dtype,
    )
    hidden_states.deallocate(True)
    # Same rank-6 -> rank-4 squeeze chain as gate above.
    up = ttnn.squeeze(up, 0)
    up = ttnn.squeeze(up, 1)
    up = ttnn.transpose(up, 1, 2)
    up = ttnn.squeeze(up, 1)
    up = ttnn.add(up, weights.up_proj_bias, output_tensor=up)

    # Apply SwiGLU activation (consumes gate and up internally)
    down_input = apply_swiglu(gate, up, config)
    # down_input is [B, num_experts, I]. The down matmul uses is_input_a_sparse=True,
    # where the sparsity tensor maps over A's batch dims (everything but the last 2). To
    # match our [B, num_experts] sparsity, A must be [B, num_experts, M=1, I] so that
    # batch_length_A == B*num_experts == the sparsity volume.
    down_input = ttnn.reshape(
        down_input, (batch_size, config.num_experts, seq_len, weights.intermediate_size_per_device)
    )
    # Down projection
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        # nnz intentionally omitted (None -> inferred at runtime). Passing a static
        # nnz makes the sparse_matmul in0-mcast receivers loop a fixed count while the
        # sender only mcasts for the *actual* non-zero `sparsity` entries. The decode
        # routing weights (softmax over top-k, scattered) frequently have <k non-zeros
        # on Blackhole (small weights flush to 0), so a static nnz != actual count and
        # the receivers deadlock in noc_semaphore_wait. Inferring the count is robust.
        # See tenstorrent/tt-metal#45943 (op deadlock) / #45052 (gpt-oss hang).
        nnz=None,
        memory_config=matmul_mem_config,
        output_tile=output_tile,
        is_input_a_sparse=True,
        # The default flips is_input_b_sparse to True as well, which makes the kernel
        # ignore A's batch dims. For B>1 the sparsity must span [B, num_experts] =
        # batch_length_A, so B must be declared dense.
        is_input_b_sparse=False,
        program_config=program_config.get_decode_down_config(
            down_input.shape[2], weights.down_proj.shape[-1], k=down_input.shape[-1]
        ),
        dtype=activation_dtype,
    )

    down_input.deallocate(True)
    sparsity.deallocate(True)
    # down output is [B, num_experts, 1, hidden] from the sparse-A batched matmul above;
    # drop the M=1 dim to get [B, num_experts, hidden].
    next_states = ttnn.squeeze(down, 2)
    next_states = ttnn.add(next_states, weights.down_proj_bias, output_tensor=next_states)
    # routing_weights arrives as [B, num_experts]. The previous permute(1, 0) was a no-op
    # at B=1 but reorders elements for B>1; we just need [B, num_experts, 1].
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
