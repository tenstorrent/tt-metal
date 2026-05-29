# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
GDN (Gated DeltaNet) prefill kernel dispatcher for Qwen3.5-9B (single device).

Adapted from the Qwen3.5-27B multi-device implementation. Supports only
single-device operation (no TP / mesh).

The prefill kernel processes N tokens per head-pair in a single dispatch.
State stays in L1 across all tokens — loaded once, written once.

Recurrence math (per token):
  1. L2-norm Q, scale by Dk^-0.5
  2. L2-norm K
  3. beta = sigmoid(b), g = -exp(A) * softplus(a + dt_bias)
  4. state *= exp(g)
  5. kv_mem = k @ state
  6. delta = beta * (v - kv_mem)
  7. state += outer(k, delta)
  8. output = q @ state
"""


import os

from loguru import logger

import ttnn

# Kernel file paths (relative to TT_METAL_HOME)
_KERNEL_DIR = "models/demos/blackhole/qwen3_5_9b/tt/gdn_kernel/kernels"
READER_PREFILL_PATH = f"{_KERNEL_DIR}/dataflow/reader_gdn_prefill.cpp"
WRITER_PREFILL_PATH = f"{_KERNEL_DIR}/dataflow/writer_gdn_prefill.cpp"
COMPUTE_PREFILL_PATH = f"{_KERNEL_DIR}/compute/gdn_prefill.cpp"

# Decode (recurrence) kernel paths — imported from the Qwen3.5-27B branch.
# Single-token recurrence kernel that replaces the slow ttnn-ops decode path.
# TensorAccessor reader/writer (no IAF variant on this branch).
READER_RECURRENCE_PATH = f"{_KERNEL_DIR}/dataflow/reader_gdn.cpp"
WRITER_RECURRENCE_PATH = f"{_KERNEL_DIR}/dataflow/writer_gdn.cpp"
COMPUTE_RECURRENCE_PATH = f"{_KERNEL_DIR}/compute/gdn_recurrence.cpp"

# Tile constants (Dk=128, Dv=128 → 128/32 = 4 tiles each)
Kt = 4
Vt = 4
STATE_TILES = Kt * Vt  # 16
BF16_TILE_BYTES = 32 * 32 * 2  # 2048

# Inner safety net: if the fused recurrence kernel raises at dispatch time, fall
# back to standard ttnn ops (on the same kernel-layout tensors). Independent of
# the module-level QWEN9B_GDN_DECODE_KERNEL flag, which decides whether the
# kernel path is taken at all.
_recurrence_fused_available = not os.environ.get("GDN_DISABLE_FUSED", "")


def _make_cb(cb_index, num_tiles, core_range_set, data_format=ttnn.bfloat16):
    """Create a CBDescriptor."""
    page_size = BF16_TILE_BYTES
    fmt = ttnn.CBFormatDescriptor(
        buffer_index=cb_index,
        data_format=data_format,
        page_size=page_size,
    )
    return ttnn.CBDescriptor(
        total_size=num_tiles * page_size,
        core_ranges=core_range_set,
        format_descriptors=[fmt],
    )


def _build_prefill_device_program(
    conv_out_dev,
    a_dev,
    b_dev,
    neg_exp_A_dev,
    dt_bias_dev,
    norm_w_dev,
    scale_dev,
    rms_scale_dev,
    rms_eps_dev,
    state_dev,
    output_dev,
    num_pairs_total,
    num_tokens,
    num_cores,
    grid,
    state_in_l1=False,
    Nv_TP=32,
    Nk_TP=16,
    repeat_factor=2,
    key_dim_tp=2048,
    v_split=1,
):
    """Build ProgramDescriptor for the prefill GDN kernel.

    Args:
        conv_out_dev: [1, N, qkv_dim] — all tokens post-conv+silu
        a_dev, b_dev: [1, N, Nv] — gate inputs for all tokens
        neg_exp_A_dev, dt_bias_dev: [1, 1, Nv] — constants
        norm_w_dev: [1, 1, Dv] — RMS norm weight
        scale_dev, rms_scale_dev, rms_eps_dev: [1,1,1] scalars
        state_dev: [num_pairs, Dk, Dv] — recurrence state
        output_dev: [num_pairs * N, 1, Dv] — flat output buffer
        v_split: number of V-shards per pair. Each (pair, v_shard_idx) tuple
            is assigned to one core; total active cores = num_pairs * v_split.
            Vt_shard = Vt // v_split tiles per shard.
    """
    assert Vt % v_split == 0, f"Vt={Vt} must be divisible by v_split={v_split}"
    Vt_shard = Vt // v_split
    state_tiles_shard = Kt * Vt_shard

    max_cores = grid.x * grid.y
    total_work_units = num_pairs_total * v_split
    num_cores_used = min(total_work_units, max_cores)
    assert (
        num_cores_used == total_work_units
    ), f"v_split={v_split} requires {total_work_units} cores; only {max_cores} available"

    # Each core gets exactly one (pair_idx, v_shard_idx) work unit.
    core_coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(num_cores_used)]
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in core_coords])

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()

    # Layout: cores 0..num_pairs-1 = (pair=0..num_pairs-1, v_shard=0)
    #         cores num_pairs..2*num_pairs-1 = (pair=0..num_pairs-1, v_shard=1)
    #         ... etc. Simple row-major over (v_shard, pair).
    for i, cc in enumerate(core_coords):
        v_shard_idx = i // num_pairs_total
        pair_idx = i % num_pairs_total
        reader_rt_args[cc.x][cc.y] = [
            conv_out_dev.buffer_address(),
            a_dev.buffer_address(),
            b_dev.buffer_address(),
            neg_exp_A_dev.buffer_address(),
            dt_bias_dev.buffer_address(),
            norm_w_dev.buffer_address(),
            scale_dev.buffer_address(),
            rms_scale_dev.buffer_address(),
            state_dev.buffer_address(),
            rms_eps_dev.buffer_address(),
            pair_idx,
            1,  # n_pairs per core (always 1 with v_split dispatch)
            v_shard_idx,
        ]
        writer_rt_args[cc.x][cc.y] = [
            output_dev.buffer_address(),
            state_dev.buffer_address(),
            pair_idx,
            1,
            v_shard_idx,
        ]

    # Tile offsets for Q/K/V regions within conv_out
    key_tile_off = key_dim_tp // 32
    v_tile_off = 2 * key_tile_off
    qkv_dim = key_dim_tp * 2 + Nv_TP * 128  # q_dim + k_dim + v_dim
    conv_tiles_per_row = qkv_dim // 32
    ab_tiles_per_row = (Nv_TP + 31) // 32

    # CB descriptors. CBs that touch only the V-shard slice are sized by Vt_shard;
    # CBs over Q/K dims (or shared scalars) are unchanged.
    cb_descriptors = [
        _make_cb(0, Kt, core_ranges),  # cb_q_raw
        _make_cb(1, Kt, core_ranges),  # cb_k_raw
        _make_cb(2, Kt, core_ranges),  # cb_k_col
        _make_cb(3, Vt_shard, core_ranges),  # cb_v
        _make_cb(4, 1, core_ranges),  # cb_g
        _make_cb(5, 1, core_ranges),  # cb_beta
        _make_cb(6, state_tiles_shard, core_ranges),  # cb_state_in
        _make_cb(7, state_tiles_shard, core_ranges),  # cb_state_b
        _make_cb(8, state_tiles_shard, core_ranges),  # cb_state_out
        _make_cb(9, 1, core_ranges),  # cb_a
        _make_cb(10, 1, core_ranges),  # cb_b
        _make_cb(12, 1, core_ranges),  # cb_neg_exp_A
        _make_cb(13, 1, core_ranges),  # cb_dt_bias
        _make_cb(14, Vt_shard, core_ranges),  # cb_norm_w (persistent)
        _make_cb(15, 1, core_ranges),  # cb_scale (persistent)
        _make_cb(16, Vt_shard, core_ranges),  # cb_out
        _make_cb(17, Kt, core_ranges),  # cb_q (normed)
        _make_cb(18, Kt, core_ranges),  # cb_k_row (normed)
        # cb_scratch holds: Kt(Q) + Kt(K) + Vt_shard(V) + 1(A) + 1(B) + 1(scalars). Keep at 15 (safe).
        _make_cb(21, 15, core_ranges),
        _make_cb(24, 1, core_ranges),  # cb_exp_g
        _make_cb(25, Vt_shard, core_ranges),  # cb_kv_mem
        _make_cb(26, Vt_shard, core_ranges),  # cb_delta
        _make_cb(27, Vt_shard, core_ranges),  # cb_delta_s
        _make_cb(28, Kt, core_ranges),  # cb_sq_acc
        _make_cb(29, 1, core_ranges),  # cb_tmp
        _make_cb(31, 1, core_ranges),  # cb_rms_scale (persistent)
        _make_cb(19, 1, core_ranges),  # cb_reduce_scaler (persistent)
        _make_cb(20, 1, core_ranges),  # cb_rms_eps (persistent)
    ]

    state_l1_flag = 1 if state_in_l1 else 0
    packed_reduce_scaler = 0x3F803F80

    reader_ct = [
        Kt,  # 0
        Vt_shard,  # 1  (per-shard V-tile count)
        BF16_TILE_BYTES,  # 2
        state_l1_flag,  # 3
        packed_reduce_scaler,  # 4
        Nv_TP,  # 5
        Nk_TP,  # 6
        repeat_factor,  # 7
        key_tile_off,  # 8
        v_tile_off,  # 9
        num_tokens,  # 10
        conv_tiles_per_row,  # 11
        ab_tiles_per_row,  # 12
        Vt,  # 13 — global V-tile count (for addressing)
    ]

    writer_ct = [
        Kt,  # 0
        Vt_shard,  # 1  (per-shard V-tile count)
        BF16_TILE_BYTES,  # 2
        state_l1_flag,  # 3
        num_tokens,  # 4
        Vt,  # 5 — global V-tile count (for addressing)
    ]

    compute_ct = [Kt, Vt_shard, 1, num_tokens]  # 1 pair per core; Vt_shard internally

    reader_kd = ttnn.KernelDescriptor(
        kernel_source=READER_PREFILL_PATH,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_kd = ttnn.KernelDescriptor(
        kernel_source=WRITER_PREFILL_PATH,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_kd = ttnn.KernelDescriptor(
        kernel_source=COMPUTE_PREFILL_PATH,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kd, writer_kd, compute_kd],
        cbs=cb_descriptors,
    )


def gdn_prefill_fused(
    conv_out,
    a_fused,
    b_fused,
    neg_exp_A,
    dt_bias,
    norm_w,
    scale_tt,
    rms_scale_tt,
    rms_eps_tt,
    state,
    output,
    num_pairs,
    num_tokens,
    num_cores=32,
    Nv_TP=32,
    Nk_TP=16,
    repeat_factor=2,
    key_dim_tp=2048,
    v_split=1,
):
    """Prefill GDN: process N tokens in a single kernel dispatch.

    State stays in L1 across all tokens — loaded once, written once.

    Args:
        conv_out: [1, N, qkv_dim] — all tokens' post-conv+silu output
        a_fused: [1, N, Nv] — gate input a for all tokens
        b_fused: [1, N, Nv] — gate input b for all tokens
        neg_exp_A: [1, 1, Nv] — precomputed -exp(A_log)
        dt_bias: [1, 1, Nv] — dt_bias constant
        norm_w: [1, 1, Dv] — RMS norm weight (persistent constant for reader)
        scale_tt: [1, 1, 1] — Q scale (Dk^-0.5)
        rms_scale_tt: [1, 1, 1] — sqrt(Dv)
        rms_eps_tt: [1, 1, 1] — Dv * eps
        state: [num_pairs, Dk, Dv] — recurrence state (updated in-place)
        output: [num_pairs * N, 1, Dv] — flat output buffer
        num_pairs: B * Nv (e.g. 1 * 32 = 32)
        num_tokens: N (sequence length)
        v_split: number of V-shards per pair (default 1). With v_split=k, each pair's
            V-tiles (Vt) are split across k cores; total active cores = num_pairs * k.
            Math is unchanged (v_split is parallelism only); output PCC must be parity.
    """
    device = conv_out.device()
    state_in_l1 = state.memory_config().buffer_type == ttnn.BufferType.L1

    # Single device path
    all_tensors = [
        conv_out,
        a_fused,
        b_fused,
        neg_exp_A,
        dt_bias,
        norm_w,
        scale_tt,
        rms_scale_tt,
        rms_eps_tt,
        state,
        output,
    ]
    devs = [ttnn.get_device_tensors(t)[0] for t in all_tensors]
    grid = devs[0].device().compute_with_storage_grid_size()

    program = _build_prefill_device_program(
        *devs,
        num_pairs,
        num_tokens,
        num_cores,
        grid,
        state_in_l1=state_in_l1,
        Nv_TP=Nv_TP,
        Nk_TP=Nk_TP,
        repeat_factor=repeat_factor,
        key_dim_tp=key_dim_tp,
        v_split=v_split,
    )
    ttnn.generic_op(all_tensors, program)


# ===========================================================================
# Decode (single-token) recurrence kernel — imported from Qwen3.5-27B.
#
# Replaces the slow ttnn-ops decode recurrence with a single fused dispatch.
# State stays per-pair in L1 CBs (ping-pong) during the dispatch and is written
# back in place. Math per pair:
#   1. state *= exp(g)              -- decay
#   2. kv_mem = k_row @ state       -- [1,K] x [K,V] -> [1,V]
#   3. delta  = beta * (v - kv_mem) -- element-wise
#   4. state += outer(k_col, delta) -- [K,1] x [1,V] -> [K,V]
#   5. output = q @ state           -- [1,K] x [K,V] -> [1,V]
# q must be pre-scaled and L2-normed; k L2-normed; g pre-exp; beta post-sigmoid.
# ===========================================================================


def _build_recurrence_device_program(
    q_dev,
    k_row_dev,
    k_col_dev,
    v_dev,
    g_dev,
    beta_dev,
    state_dev,
    output_dev,
    state_out_dev,
    num_pairs_total,
    num_cores,
    grid,
):
    """Build a single-device ProgramDescriptor for the GDN recurrence kernel.

    Each core processes a contiguous run of pairs. Tensors must be bfloat16,
    TILE_LAYOUT, interleaved DRAM (TensorAccessor reader/writer).

    Args:
        q_dev, k_row_dev: [num_pairs, 1, Dk] (Kt tiles each)
        k_col_dev: [num_pairs, Dk, 1] (Kt tiles)
        v_dev: [num_pairs, 1, Dv] (Vt tiles)
        g_dev, beta_dev: [num_pairs, 1, 1] (1 tile each)
        state_dev: [num_pairs, Dk, Dv] (Kt*Vt tiles) — recurrence state in
        output_dev: [num_pairs, 1, Dv] (Vt tiles)
        state_out_dev: same buffer as state_dev for in-place update
    """
    max_cores = grid.x * grid.y
    num_cores = min(num_cores, num_pairs_total, max_cores)
    pairs_per_core = num_pairs_total // num_cores
    remainder = num_pairs_total % num_cores

    core_coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(num_cores)]
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in core_coords])

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    pair_offset = 0
    core_pair_counts = []
    for i, cc in enumerate(core_coords):
        n = pairs_per_core + (1 if i < remainder else 0)
        core_pair_counts.append(n)
        reader_rt_args[cc.x][cc.y] = [
            q_dev.buffer_address(),
            k_row_dev.buffer_address(),
            k_col_dev.buffer_address(),
            v_dev.buffer_address(),
            g_dev.buffer_address(),
            beta_dev.buffer_address(),
            state_dev.buffer_address(),
            pair_offset,
            n,
        ]
        writer_rt_args[cc.x][cc.y] = [
            output_dev.buffer_address(),
            state_out_dev.buffer_address(),
            pair_offset,
            n,
        ]
        pair_offset += n

    cb_descriptors = [
        _make_cb(0, Kt, core_ranges),  # cb_q
        _make_cb(1, Kt, core_ranges),  # cb_k_row
        _make_cb(2, Kt, core_ranges),  # cb_k_col
        _make_cb(3, Vt, core_ranges),  # cb_v
        _make_cb(4, 1, core_ranges),  # cb_g
        _make_cb(5, 1, core_ranges),  # cb_beta
        _make_cb(6, STATE_TILES, core_ranges),  # cb_state_in (reader fills)
        _make_cb(7, STATE_TILES, core_ranges),  # cb_state_b (decayed state)
        _make_cb(8, STATE_TILES, core_ranges),  # cb_state_out (writer reads)
        _make_cb(16, Vt, core_ranges),  # cb_out
        _make_cb(24, 1, core_ranges),  # cb_exp_g
        _make_cb(25, Vt, core_ranges),  # cb_kv_mem
        _make_cb(26, Vt, core_ranges),  # cb_delta (v - kv_mem)
        _make_cb(27, Vt, core_ranges),  # cb_delta_s (beta * delta)
    ]

    # Group cores by pair count so the compute kernel gets the right per-core
    # pair count as a compile-time arg.
    groups = {}
    for i, cc in enumerate(core_coords):
        groups.setdefault(core_pair_counts[i], []).append(cc)

    all_kernels = []
    for n_pairs, cores in groups.items():
        if n_pairs == 0:
            continue
        group_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

        group_reader_rt = ttnn.RuntimeArgs()
        group_writer_rt = ttnn.RuntimeArgs()
        for c in cores:
            group_reader_rt[c.x][c.y] = list(reader_rt_args[c.x][c.y])
            group_writer_rt[c.x][c.y] = list(writer_rt_args[c.x][c.y])

        reader_ct = [Kt, Vt, BF16_TILE_BYTES]
        for t in [q_dev, k_row_dev, k_col_dev, v_dev, g_dev, beta_dev, state_dev]:
            reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

        writer_ct = [Kt, Vt, BF16_TILE_BYTES]
        writer_ct.extend(ttnn.TensorAccessorArgs(output_dev).get_compile_time_args())
        writer_ct.extend(ttnn.TensorAccessorArgs(state_out_dev).get_compile_time_args())

        reader_kd = ttnn.KernelDescriptor(
            kernel_source=READER_RECURRENCE_PATH,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=group_ranges,
            compile_time_args=reader_ct,
            runtime_args=group_reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer_kd = ttnn.KernelDescriptor(
            kernel_source=WRITER_RECURRENCE_PATH,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=group_ranges,
            compile_time_args=writer_ct,
            runtime_args=group_writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        )
        compute_kd = ttnn.KernelDescriptor(
            kernel_source=COMPUTE_RECURRENCE_PATH,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=group_ranges,
            compile_time_args=[Kt, Vt, n_pairs],
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=False,
            ),
        )
        all_kernels.extend([reader_kd, writer_kd, compute_kd])

    return ttnn.ProgramDescriptor(kernels=all_kernels, cbs=cb_descriptors)


def _gdn_recurrence_fused(q, k_row, k_col, v, g, beta, state, output, state_out, num_cores=32):
    """Dispatch the fused GDN recurrence kernel (single device)."""
    num_pairs_total = q.shape[0]
    all_tensors = [q, k_row, k_col, v, g, beta, state, output, state_out]
    devs = [ttnn.get_device_tensors(t)[0] for t in all_tensors]
    grid = devs[0].device().compute_with_storage_grid_size()
    program = _build_recurrence_device_program(*devs, num_pairs_total, num_cores, grid)
    return ttnn.generic_op(all_tensors, program)


def _gdn_recurrence_ttnn(q, k_row, k_col, v, g, beta, state):
    """GDN recurrence step using standard ttnn ops (fallback for the fused kernel).

    Operates on the same kernel-layout tensors: q/k_row [P,1,K], k_col [P,K,1],
    v [P,1,V], g/beta [P,1,1], state [P,K,V]. Updates state in place.
    """
    g_exp = ttnn.exp(g)
    state_b = ttnn.multiply(state, g_exp)
    ttnn.deallocate(g_exp)

    kv_mem = ttnn.matmul(k_row, state_b)
    diff = ttnn.subtract(v, kv_mem)
    ttnn.deallocate(kv_mem)
    delta = ttnn.multiply(beta, diff)
    ttnn.deallocate(diff)

    outer = ttnn.matmul(k_col, delta)
    ttnn.deallocate(delta)
    new_state = ttnn.add(state_b, outer)
    ttnn.deallocate(state_b)
    ttnn.deallocate(outer)

    output = ttnn.matmul(q, new_state)

    ttnn.copy(new_state, state)
    ttnn.deallocate(new_state)
    return output


def gdn_recurrence_fused_inplace(q, k_row, k_col, v, g, beta, state, output, num_cores=32):
    """Compute the GDN recurrence and write the result into `output` (in place).

    State is updated in place (the kernel writer writes back to the `state`
    buffer address). Tries the fused kernel first; falls back to ttnn ops if
    the kernel is disabled or raises.
    """
    global _recurrence_fused_available

    if _recurrence_fused_available:
        try:
            # Prime DRAM: touch an input via standard ttnn so writes from prior
            # ops are visible to the custom kernel's NOC reads. Workaround for
            # Blackhole DRAM coherence with noc_async_read_page.
            tmp = ttnn.add(q, 0.0)
            ttnn.deallocate(tmp)
            _gdn_recurrence_fused(q, k_row, k_col, v, g, beta, state, output, state, num_cores)
            return
        except Exception as e:
            logger.warning(f"Fused GDN recurrence kernel failed, falling back to ttnn ops: {e}")
            _recurrence_fused_available = False

    result = _gdn_recurrence_ttnn(q, k_row, k_col, v, g, beta, state)
    ttnn.copy(result, output)
    ttnn.deallocate(result)
