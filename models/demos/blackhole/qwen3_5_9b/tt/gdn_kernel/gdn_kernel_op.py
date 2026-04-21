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


import ttnn

# Kernel file paths (relative to TT_METAL_HOME)
_KERNEL_DIR = "models/demos/blackhole/qwen3_5_9b/tt/gdn_kernel/kernels"
READER_PREFILL_PATH = f"{_KERNEL_DIR}/dataflow/reader_gdn_prefill.cpp"
WRITER_PREFILL_PATH = f"{_KERNEL_DIR}/dataflow/writer_gdn_prefill.cpp"
COMPUTE_PREFILL_PATH = f"{_KERNEL_DIR}/compute/gdn_prefill.cpp"

# Tile constants (Dk=128, Dv=128 → 128/32 = 4 tiles each)
Kt = 4
Vt = 4
STATE_TILES = Kt * Vt  # 16
BF16_TILE_BYTES = 32 * 32 * 2  # 2048


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
    """
    max_cores = grid.x * grid.y
    num_cores = min(num_cores, num_pairs_total, max_cores)
    pairs_per_core = num_pairs_total // num_cores
    remainder = num_pairs_total % num_cores

    core_coords = []
    for i in range(num_cores):
        core_coords.append(ttnn.CoreCoord(i % grid.x, i // grid.x))

    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in core_coords])

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    pair_offset = 0
    core_pair_counts = []

    for i, cc in enumerate(core_coords):
        n = pairs_per_core + (1 if i < remainder else 0)
        core_pair_counts.append(n)
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
            pair_offset,
            n,
        ]
        writer_rt_args[cc.x][cc.y] = [
            output_dev.buffer_address(),
            state_dev.buffer_address(),
            pair_offset,
            n,
        ]
        pair_offset += n

    # Tile offsets for Q/K/V regions within conv_out
    key_tile_off = key_dim_tp // 32
    v_tile_off = 2 * key_tile_off
    qkv_dim = key_dim_tp * 2 + Nv_TP * 128  # q_dim + k_dim + v_dim
    conv_tiles_per_row = qkv_dim // 32
    ab_tiles_per_row = (Nv_TP + 31) // 32

    # CB descriptors
    cb_descriptors = [
        _make_cb(0, Kt, core_ranges),  # cb_q_raw
        _make_cb(1, Kt, core_ranges),  # cb_k_raw
        _make_cb(2, Kt, core_ranges),  # cb_k_col
        _make_cb(3, Vt, core_ranges),  # cb_v
        _make_cb(4, 1, core_ranges),  # cb_g
        _make_cb(5, 1, core_ranges),  # cb_beta
        _make_cb(6, STATE_TILES, core_ranges),  # cb_state_in
        _make_cb(7, STATE_TILES, core_ranges),  # cb_state_b
        _make_cb(8, STATE_TILES, core_ranges),  # cb_state_out
        _make_cb(9, 1, core_ranges),  # cb_a
        _make_cb(10, 1, core_ranges),  # cb_b
        _make_cb(12, 1, core_ranges),  # cb_neg_exp_A
        _make_cb(13, 1, core_ranges),  # cb_dt_bias
        _make_cb(14, Vt, core_ranges),  # cb_norm_w (persistent)
        _make_cb(15, 1, core_ranges),  # cb_scale (persistent)
        _make_cb(16, Vt, core_ranges),  # cb_out
        _make_cb(17, Kt, core_ranges),  # cb_q (normed)
        _make_cb(18, Kt, core_ranges),  # cb_k_row (normed)
        _make_cb(21, 15, core_ranges),  # cb_scratch (whole-tile Q/K/V/a/b + scalars)
        _make_cb(24, 1, core_ranges),  # cb_exp_g
        _make_cb(25, Vt, core_ranges),  # cb_kv_mem
        _make_cb(26, Vt, core_ranges),  # cb_delta
        _make_cb(27, Vt, core_ranges),  # cb_delta_s
        _make_cb(28, Kt, core_ranges),  # cb_sq_acc
        _make_cb(29, 1, core_ranges),  # cb_tmp
        _make_cb(31, 1, core_ranges),  # cb_rms_scale (persistent)
        _make_cb(19, 1, core_ranges),  # cb_reduce_scaler (persistent)
        _make_cb(20, 1, core_ranges),  # cb_rms_eps (persistent)
    ]

    state_l1_flag = 1 if state_in_l1 else 0
    packed_reduce_scaler = 0x3F803F80

    # Group cores by pair count (for compile-time arg specialization)
    groups = {}
    for i, cc in enumerate(core_coords):
        n = core_pair_counts[i]
        groups.setdefault(n, []).append(cc)

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

        reader_ct = [
            Kt,  # 0
            Vt,  # 1
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
        ]

        writer_ct = [
            Kt,  # 0
            Vt,  # 1
            BF16_TILE_BYTES,  # 2
            state_l1_flag,  # 3
            num_tokens,  # 4
        ]

        reader_kd = ttnn.KernelDescriptor(
            kernel_source=READER_PREFILL_PATH,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=group_ranges,
            compile_time_args=reader_ct,
            runtime_args=group_reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        )

        writer_kd = ttnn.KernelDescriptor(
            kernel_source=WRITER_PREFILL_PATH,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=group_ranges,
            compile_time_args=writer_ct,
            runtime_args=group_writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        )

        compute_kd = ttnn.KernelDescriptor(
            kernel_source=COMPUTE_PREFILL_PATH,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=group_ranges,
            compile_time_args=[Kt, Vt, n_pairs, num_tokens],
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=False,
            ),
        )

        all_kernels.extend([reader_kd, writer_kd, compute_kd])

    return ttnn.ProgramDescriptor(
        kernels=all_kernels,
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
    )
    ttnn.generic_op(all_tensors, program)
