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


class DenseDecodePlan:
    """Memory and program configs of decode_forward_dense for one (device grid, shapes) combination."""

    def __init__(self, mesh_device, hidden, intermediate_local, num_experts, tokens):
        from ..fused_decode.config import DENSE_MOE_CORES, DENSE_MOE_ELTWISE_CORES, DENSE_MOE_WORKERS_PER_BANK

        tile = ttnn.TILE_SIZE
        self.tokens = tokens
        grid = mesh_device.compute_with_storage_grid_size()
        cores = DENSE_MOE_CORES
        n_act = num_experts * intermediate_local
        assert hidden % (tile * cores) == 0 and n_act % (tile * cores) == 0, (hidden, n_act, cores)
        core_set = ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True)

        def width_sharded(cols, cores_used=core_set):
            return ttnn.create_sharded_memory_config(
                (tokens, cols),
                cores_used,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        def dram_sharded(k_tiles_per_core, per_core_n):
            return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=max(d for d in range(1, min(k_tiles_per_core, 16) + 1) if k_tiles_per_core % d == 0),
                per_core_M=tokens // tile,
                per_core_N=per_core_n,
                fused_activation=None,
                num_workers_per_dram_bank=DENSE_MOE_WORKERS_PER_BANK,
            )

        def mcast_1d(per_core_n, n_cores=cores):
            # [tokens, num_experts] routing weights times a [num_experts, n] constant, written straight into the
            # width-sharded layout of the tensor it scales / is added to.
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, -(-n_cores // grid.x)),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=per_core_n,
                per_core_M=tokens // tile,
                per_core_N=per_core_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )

        self.input_memory_config = width_sharded(hidden // cores)
        self.act_memory_config = width_sharded(n_act // cores)
        self.out_memory_config = width_sharded(hidden // cores)
        self.gate_up_program_config = dram_sharded(hidden // tile // cores, n_act // tile // cores)
        self.down_program_config = dram_sharded(n_act // tile // cores, hidden // tile // cores)
        ew = DENSE_MOE_ELTWISE_CORES
        assert n_act % (tile * ew) == 0, (n_act, ew)
        self.eltwise_memory_config = width_sharded(n_act // ew, ttnn.num_cores_to_corerangeset(ew, grid, row_wise=True))
        self.expand_program_config = mcast_1d(n_act // tile // ew, ew)
        self.down_bias_program_config = mcast_1d(hidden // tile // cores)
        # LoFi like the sparse matmuls it replaces (BFP8 activations x BFP4 weights).
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )


def decode_forward_dense(hidden_states, routing_weights, weights, plan: DenseDecodePlan, config: ExpertConfig):
    """Decode MoE for one 32-token tile per device over every expert (gpt-oss-20b batch 32, TP over the intermediate
    dimension, no expert parallelism).

    hidden_states: [1, 1, 32, hidden] (the post-attention norm output), replicated over TP.
    routing_weights: [32, num_experts] router output (softmax over each token's top-k, zeros elsewhere).
    weights: DenseDecodeExpertWeights (every expert's TP shard side by side).

    gate = x @ gate_all + b, up = x @ up_all + b ([32, E * I_local], width-sharded); the gpt-oss SwiGLU as one
    multiply; each expert's columns scaled by the token's routing weight; then one down matmul over all experts, whose
    K dimension sums the weighted experts: sum_e w[t, e] (act_e[t] @ down_e) = (w * act)[t] @ down_all. The down bias
    term sum_e w[t, e] b_e is routing_weights @ down_bias. Returns this device's partial sum [1, 1, 32, hidden]
    (DRAM interleaved, BF8, as the prefill path hands it over: ttnn.all_reduce misreads some L1 BF8 buffers, #41640)
    for the TP all-reduce.
    """
    UWP, U = ttnn.UnaryWithParam, ttnn.UnaryOpType
    limit, alpha = config.swiglu_limit, config.alpha
    ck = plan.compute_kernel_config
    x = ttnn.to_memory_config(hidden_states, plan.input_memory_config, dtype=ttnn.bfloat8_b)
    gate = ttnn.matmul(
        x,
        weights.gate,
        program_config=plan.gate_up_program_config,
        memory_config=plan.act_memory_config,
        dtype=ttnn.bfloat16,
        compute_kernel_config=ck,
    )
    up = ttnn.matmul(
        x,
        weights.up,
        program_config=plan.gate_up_program_config,
        memory_config=plan.act_memory_config,
        dtype=ttnn.bfloat16,
        compute_kernel_config=ck,
    )
    x.deallocate(True)
    # The elementwise steps run on twice the matmul's cores (plan.eltwise_memory_config).
    ew = plan.eltwise_memory_config
    gate_ew = ttnn.to_memory_config(gate, ew)
    gate.deallocate(True)
    up_ew = ttnn.to_memory_config(up, ew)
    up.deallocate(True)
    gate = ttnn.add(gate_ew, weights.gate_bias, memory_config=ew, output_tensor=gate_ew)
    up = ttnn.add(up_ew, weights.up_bias, memory_config=ew, output_tensor=up_ew)
    w = ttnn.to_memory_config(routing_weights, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    w = ttnn.reshape(w, (1, 1, w.shape[-2], w.shape[-1]))
    w_cols = ttnn.matmul(
        w,
        weights.expand,
        program_config=plan.expand_program_config,
        memory_config=ew,
        dtype=ttnn.bfloat16,
    )
    # (clamp(up, -l, l) + 1) * g * sigmoid(alpha * g), g = clamp(gate, max=l); g * sigmoid(alpha g) = silu(alpha g) / alpha.
    act = ttnn.multiply(
        up,
        gate,
        input_tensor_a_activations=[UWP(U.CLAMP_TSS, -limit, limit), UWP(U.ADD_UNARY_SFPU, 1.0)],
        input_tensor_b_activations=[
            UWP(U.CLAMP_TSS, -3.0e38, limit),
            UWP(U.MUL_UNARY_SFPU, alpha),
            UWP(U.SILU),
            UWP(U.MUL_UNARY_SFPU, 1.0 / alpha),
        ],
        memory_config=ew,
    )
    gate.deallocate(True)
    up.deallocate(True)
    act = ttnn.multiply(act, w_cols, memory_config=ew, output_tensor=act)
    w_cols.deallocate(True)
    act_ew = act
    act = ttnn.to_memory_config(act_ew, plan.act_memory_config)
    act_ew.deallocate(True)
    down = ttnn.matmul(
        act,
        weights.down,
        program_config=plan.down_program_config,
        memory_config=plan.out_memory_config,
        dtype=ttnn.bfloat16,
        compute_kernel_config=ck,
    )
    act.deallocate(True)
    bias = ttnn.matmul(
        w,
        weights.down_bias,
        program_config=plan.down_bias_program_config,
        memory_config=plan.out_memory_config,
        dtype=ttnn.bfloat16,
    )
    w.deallocate(True)
    down = ttnn.add(down, bias, memory_config=plan.out_memory_config, output_tensor=down)
    bias.deallocate(True)
    out = ttnn.to_memory_config(down, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    down.deallocate(True)
    return out


def decode_forward_stream(hidden_states, expert_indices, expert_weights, weights, stream, send=None):
    """Decode MoE for one token over its top-k experts with the DRAM-streaming ops of fused_decode/stream.py.

    hidden_states: the flat BF16 norm output of the layer boundary (fused_decode/boundary.py, one token).
    expert_indices / expert_weights: the streamed router's [1, 32] UINT16 ids / BF16 softmax weights buffers (top-k
    in the first k entries; shared, not deallocated here).
    gate|up + bias + SwiGLU + routing-weight scaling run in one op into the shared [k, I_pad] activation; the down
    projection + bias + sum over the routed experts run in a second op, straight into this device's flat partial sum.
    Returns that partial sum (the layer's next boundary all-reduces it and adds it to the residual); send: fuse the
    boundary's fabric send into the down op (ccl.CCLManager.decode_boundary_send).
    """
    act = stream["gate_up"](hidden_states, expert_indices, expert_weights, weights.gate_up_stream, stream["act"])
    return stream["down"](act, expert_indices, expert_weights, weights.down_stream, stream["partial"], send=send)
