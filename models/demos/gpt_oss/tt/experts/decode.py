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
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(
            hidden_states.shape[2], weights.gate_proj.shape[3], k=hidden_states.shape[-1]
        ),
        dtype=activation_dtype,
    )
    # Note: reshape/transpose operations return views - do not deallocate originals
    gate = ttnn.reshape(gate, (batch_size, config.num_experts, 1, weights.intermediate_size_per_device))
    gate = ttnn.transpose(gate, 1, 2)
    gate = ttnn.reshape(gate, (batch_size, config.num_experts, weights.intermediate_size_per_device))
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
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(
            hidden_states.shape[2], weights.up_proj.shape[3], k=hidden_states.shape[-1]
        ),
        dtype=activation_dtype,
    )
    hidden_states.deallocate(True)
    # Note: reshape/transpose operations return views - do not deallocate originals
    up = ttnn.reshape(up, (batch_size, config.num_experts, 1, weights.intermediate_size_per_device))
    up = ttnn.transpose(up, 1, 2)
    up = ttnn.reshape(up, (batch_size, config.num_experts, weights.intermediate_size_per_device))
    up = ttnn.add(up, weights.up_proj_bias, output_tensor=up)

    # Apply SwiGLU activation (consumes gate and up internally)
    down_input = apply_swiglu(gate, up, config)
    # Note: transpose/reshape operations return views - do not deallocate originals
    down_input = ttnn.transpose(down_input, 1, 0)
    down_input = ttnn.reshape(down_input, (1, config.num_experts, seq_len, weights.intermediate_size_per_device))
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
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        is_input_a_sparse=True,
        program_config=program_config.get_decode_down_config(
            down_input.shape[2], weights.down_proj.shape[-1], k=down_input.shape[-1]
        ),
        dtype=activation_dtype,
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


def _indexed_program_config(grid, n, k, in0_block_w):
    """1D mcast-in0 config for the indexed sparse_matmul: every core of `grid` owns per_core_N output tiles."""
    cores = grid[0] * grid[1]
    n_tiles, k_tiles = -(-n // ttnn.TILE_SIZE), -(-k // ttnn.TILE_SIZE)
    per_core_N = -(-n_tiles // cores)
    assert -(-n_tiles // per_core_N) == cores, f"{n_tiles} output tiles do not cover the {grid} grid"
    in0_block_w = max(d for d in range(1, in0_block_w + 1) if k_tiles % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=1,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def swiglu_fused(gate, up, config: ExpertConfig, memory_config=ttnn.L1_MEMORY_CONFIG):
    """GPT-OSS SwiGLU (clamp(up, -l, l) + 1) * g * sigmoid(alpha * g), g = clamp(gate, max=l), as one binary op:
    g * sigmoid(alpha * g) == silu(alpha * g) / alpha, applied as input activations of the multiply."""
    UWP, U = ttnn.UnaryWithParam, ttnn.UnaryOpType
    limit, alpha = config.swiglu_limit, config.alpha
    return ttnn.multiply(
        up,
        gate,
        input_tensor_a_activations=[UWP(U.CLAMP_TSS, -limit, limit), UWP(U.ADD_UNARY_SFPU, 1.0)],
        input_tensor_b_activations=[
            UWP(U.CLAMP_TSS, -3.0e38, limit),
            UWP(U.MUL_UNARY_SFPU, alpha),
            UWP(U.SILU),
            UWP(U.MUL_UNARY_SFPU, 1.0 / alpha),
        ],
        memory_config=memory_config,
    )


def decode_forward_indexed(
    hidden_states,
    expert_indices,
    expert_weights,
    weights,
    config: ExpertConfig,
    mesh_config,
    ccl_manager,
    sparsity_placeholder,
    expert_mapping,
):
    """Decode MoE for one token over its top-k experts only (indexed sparse_matmul, per-expert bias fused).

    hidden_states: [1, 1, 1 (32), hidden] BF16, L1 interleaved, replicated over TP.
    expert_indices: [1, k] UINT16 row-major (the routed expert ids).
    expert_weights: [1, 1, 1, k] row-major routing weights (softmax over the top-k logits).
    Returns the all-reduced BF16 MoE output in the decode residual layout.
    """
    if hidden_states.shape[-2] != 1:
        raise ValueError(f"Indexed decode routes one token per device, got {hidden_states.shape[-2]}")
    k = config.num_experts_per_tok
    inter = weights.intermediate_padded
    hidden = config.hidden_size
    gate_up_cfg = _indexed_program_config((8, 6), 2 * inter, hidden, 30)
    down_cfg = _indexed_program_config((10, 9), hidden, inter, 12)

    def project(x, w, b, cfg, a_sparse=False, dtype=ttnn.bfloat16):
        return ttnn.sparse_matmul(
            x,
            w,
            sparsity=sparsity_placeholder,
            indices=expert_indices,
            bias=b,
            is_input_a_sparse=a_sparse,
            program_config=cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=dtype,
        )

    # [1, 1, 1, k, 32, 2 * inter]: one tile row block per routed expert, [gate | up] along the width.
    rows = hidden_states.shape[-2]
    gate_up = project(hidden_states, weights.gate_up_proj, weights.gate_up_proj_bias, gate_up_cfg)
    gate_up = ttnn.reshape(gate_up, (1, k, rows, 2 * inter), (1, k, 32, 2 * inter))
    gate = ttnn.slice(gate_up, [0, 0, 0, 0], [1, k, rows, inter])
    up = ttnn.slice(gate_up, [0, 0, 0, inter], [1, k, rows, 2 * inter])
    gate_up.deallocate(True)
    act = swiglu_fused(gate, up, config)
    gate.deallocate(True)
    up.deallocate(True)
    # [1, k, 32, hidden] BF8: the all-reduce payload precision, so the combine below emits it directly.
    down = project(act, weights.down_proj, weights.down_proj_bias, down_cfg, a_sparse=True, dtype=ttnn.bfloat8_b)
    act.deallocate(True)

    # Routing-weighted sum over the k expert rows in one op (score multiply-accumulate inside the reduce), written
    # straight into the all-reduce layout. Scores are one row-major [tokens, 1, 1, k] row per token.
    all_reduce = ccl_manager.get_decode_all_reduce(hidden, mesh_config.tp_axis)
    scores = expert_weights
    down = ttnn.reshape(down, (k, 1, rows, hidden), (k, 1, 32, hidden))
    out = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
        down,
        expert_indices,
        expert_mapping,
        0,
        split_size=hidden,
        cluster_axis=mesh_config.tp_axis,
        output_memory_config=all_reduce.memory_config,
        scores_tensor=scores,
    )[0]
    down.deallocate(True)
    scores.deallocate(True)
    return all_reduce(out, "moe")
