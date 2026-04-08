# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
GDN (Gated DeltaNet) recurrence kernel dispatcher.

Supports both single-device and mesh (multi-device TP) operation.
For mesh devices, uses MeshProgramDescriptor to build per-device programs
with correct buffer addresses for each device's tensor shards.

Attempts the fused kernel via ttnn.generic_op first (single kernel launch,
~12x faster). Falls back to standard ttnn ops if generic_op is unavailable
or fails (e.g. unsupported platform).

Recurrence math:
  1. state *= exp(g)                    -- decay
  2. kv_mem = k_row @ state             -- [1,K] x [K,V] -> [1,V]
  3. delta = beta * (v - kv_mem)        -- element-wise
  4. state += outer(k_col, delta)       -- [K,1] x [1,V] -> [K,V]
  5. output = q @ state                 -- [1,K] x [K,V] -> [1,V]
"""

import hashlib
import os

from loguru import logger

import ttnn

# Kernel file paths (relative to TT_METAL_HOME)
_KERNEL_DIR = "models/demos/qwen35_27b/tt/gdn_kernel/kernels"
READER_PATH = f"{_KERNEL_DIR}/dataflow/reader_gdn.cpp"
WRITER_PATH = f"{_KERNEL_DIR}/dataflow/writer_gdn.cpp"
READER_IAF_PATH = f"{_KERNEL_DIR}/dataflow/reader_gdn_iaf.cpp"
WRITER_IAF_PATH = f"{_KERNEL_DIR}/dataflow/writer_gdn_iaf.cpp"
COMPUTE_PATH = f"{_KERNEL_DIR}/compute/gdn_recurrence.cpp"
READER_FUSED_PATH = f"{_KERNEL_DIR}/dataflow/reader_gdn_fused.cpp"
WRITER_FUSED_PATH = f"{_KERNEL_DIR}/dataflow/writer_gdn_fused.cpp"
COMPUTE_FUSED_PATH = f"{_KERNEL_DIR}/compute/gdn_fused.cpp"

# Use InterleavedAddrGenFast (IAF) by default; set GDN_USE_TA=1 for TensorAccessor
_USE_IAF = not os.environ.get("GDN_USE_TA", "")

# Tile constants
Kt = 4  # 128 / 32
Vt = 4  # 128 / 32
STATE_TILES = Kt * Vt  # 16
BF16_TILE_BYTES = 32 * 32 * 2  # 2048

# Module-level flags
_fused_available = not os.environ.get("GDN_DISABLE_FUSED", "")
_full_fused_available = not os.environ.get("GDN_DISABLE_FULL_FUSED", "")


def _compute_kernel_hash():
    """Hash kernel source file contents to invalidate program cache on edits."""
    tt_home = os.environ.get("TT_METAL_HOME", "")
    h = hashlib.md5()
    for path in [
        READER_PATH,
        WRITER_PATH,
        READER_IAF_PATH,
        WRITER_IAF_PATH,
        COMPUTE_PATH,
        READER_FUSED_PATH,
        WRITER_FUSED_PATH,
        COMPUTE_FUSED_PATH,
    ]:
        full = os.path.join(tt_home, path)
        try:
            with open(full, "rb") as f:
                h.update(f.read())
        except FileNotFoundError:
            h.update(path.encode())
    return int(h.hexdigest(), 16) & 0x7FFFFFFFFFFFFFFF  # positive 64-bit int


_KERNEL_CONTENT_HASH = _compute_kernel_hash()


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


def _build_device_program(
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
    state_in_l1=False,
):
    """Build a ProgramDescriptor for a single device with correct buffer addresses."""
    max_cores = grid.x * grid.y
    num_cores = min(num_cores, num_pairs_total, max_cores)
    pairs_per_core = num_pairs_total // num_cores
    remainder = num_pairs_total % num_cores

    # Build core coordinates
    core_coords = []
    for i in range(num_cores):
        core_coords.append(ttnn.CoreCoord(i % grid.x, i // grid.x))

    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in core_coords])

    # Compute per-core pair assignments with per-device buffer addresses
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

    # CB descriptors (same for all cores)
    cb_descriptors = [
        _make_cb(0, Kt, core_ranges),  # cb_q
        _make_cb(1, Kt, core_ranges),  # cb_k_row
        _make_cb(2, Kt, core_ranges),  # cb_k_col
        _make_cb(3, Vt, core_ranges),  # cb_v
        _make_cb(4, 1, core_ranges),  # cb_g
        _make_cb(5, 1, core_ranges),  # cb_beta
        _make_cb(6, STATE_TILES, core_ranges),  # cb_state_in (reader fills)
        _make_cb(7, STATE_TILES, core_ranges),  # cb_state_b (intermediate decayed state)
        _make_cb(8, STATE_TILES, core_ranges),  # cb_state_out (compute fills, writer reads)
        _make_cb(16, Vt, core_ranges),  # cb_out
        _make_cb(24, 1, core_ranges),  # cb_exp_g
        _make_cb(25, Vt, core_ranges),  # cb_kv_mem
        _make_cb(26, Vt, core_ranges),  # cb_delta (v - kv_mem)
        _make_cb(27, Vt, core_ranges),  # cb_delta_s (beta * delta)
    ]

    # Group cores by pair count (for compile-time arg specialization)
    groups = {}  # num_pairs -> list of CoreCoord
    for i, cc in enumerate(core_coords):
        n = core_pair_counts[i]
        groups.setdefault(n, []).append(cc)

    all_kernels = []
    for n_pairs, cores in groups.items():
        if n_pairs == 0:
            continue

        group_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

        # Per-group runtime args
        group_reader_rt = ttnn.RuntimeArgs()
        group_writer_rt = ttnn.RuntimeArgs()
        for c in cores:
            group_reader_rt[c.x][c.y] = list(reader_rt_args[c.x][c.y])
            group_writer_rt[c.x][c.y] = list(writer_rt_args[c.x][c.y])

        if _USE_IAF:
            state_l1_flag = 1 if state_in_l1 else 0
            reader_ct = [Kt, Vt, BF16_TILE_BYTES, state_l1_flag]
            reader_src = READER_IAF_PATH
            writer_ct = [Kt, Vt, BF16_TILE_BYTES, state_l1_flag]
            writer_src = WRITER_IAF_PATH
        else:
            reader_ct = [Kt, Vt, BF16_TILE_BYTES]
            for t in [q_dev, k_row_dev, k_col_dev, v_dev, g_dev, beta_dev, state_dev]:
                reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
            reader_src = READER_PATH
            writer_ct = [Kt, Vt, BF16_TILE_BYTES]
            writer_ct.extend(ttnn.TensorAccessorArgs(output_dev).get_compile_time_args())
            writer_ct.extend(ttnn.TensorAccessorArgs(state_out_dev).get_compile_time_args())
            writer_src = WRITER_PATH

        reader_kd = ttnn.KernelDescriptor(
            kernel_source=reader_src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=group_ranges,
            compile_time_args=reader_ct,
            runtime_args=group_reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        )

        writer_kd = ttnn.KernelDescriptor(
            kernel_source=writer_src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=group_ranges,
            compile_time_args=writer_ct,
            runtime_args=group_writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        )

        compute_kd = ttnn.KernelDescriptor(
            kernel_source=COMPUTE_PATH,
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

    return ttnn.ProgramDescriptor(
        kernels=all_kernels,
        cbs=cb_descriptors,
    )


def _gdn_recurrence_fused(
    q,
    k_row,
    k_col,
    v,
    g,
    beta,
    state,
    output,
    state_out,
    num_cores=10,
):
    """
    Execute fused GDN recurrence on device using a custom kernel.

    Supports both single-device and mesh (multi-device) operation.
    For mesh devices, builds a per-device MeshProgramDescriptor so each
    device gets correct buffer addresses for its tensor shards.

    Tensors must be bfloat16, TILE_LAYOUT, interleaved.
    State can be in L1 (preferred) or DRAM; inputs/output are in DRAM.
    """
    mesh_device = q.device()
    num_pairs_total = q.shape[0]
    mesh_shape = mesh_device.shape
    num_devices = mesh_shape[0] * mesh_shape[1]

    # Detect if state is in L1 interleaved
    state_in_l1 = state.memory_config().buffer_type == ttnn.BufferType.L1

    if num_devices == 1:
        # Single device: use ProgramDescriptor directly
        device_tensors = [q, k_row, k_col, v, g, beta, state, output, state_out]
        # For single device, get_device_tensors returns a list of 1
        devs = [ttnn.get_device_tensors(t)[0] for t in device_tensors]
        grid = devs[0].device().compute_with_storage_grid_size()
        program = _build_device_program(*devs, num_pairs_total, num_cores, grid, state_in_l1=state_in_l1)
        io_tensors = [q, k_row, k_col, v, g, beta, state, output, state_out]
        return ttnn.generic_op(io_tensors, program)

    # Multi-device: build per-device programs via MeshProgramDescriptor
    mesh_tensors = [q, k_row, k_col, v, g, beta, state, output, state_out]
    per_device = [ttnn.get_device_tensors(t) for t in mesh_tensors]

    mesh_program = ttnn.MeshProgramDescriptor()

    for row in range(mesh_shape[0]):
        for col in range(mesh_shape[1]):
            device_idx = row * mesh_shape[1] + col
            coord = ttnn.MeshCoordinate(row, col)

            # Extract this device's tensors
            devs = [per_device[i][device_idx] for i in range(len(mesh_tensors))]
            grid = devs[0].device().compute_with_storage_grid_size()

            program = _build_device_program(*devs, num_pairs_total, num_cores, grid, state_in_l1=state_in_l1)
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program

    io_tensors = [q, k_row, k_col, v, g, beta, state, output, state_out]
    return ttnn.generic_op(io_tensors, mesh_program)


# ---------------------------------------------------------------------------
# ttnn ops fallback (from gdn_kernel_op_ttnn.py)
# ---------------------------------------------------------------------------


def _gdn_recurrence_ttnn(q, k_row, k_col, v, g, beta, state):
    """GDN recurrence step using standard ttnn ops (fallback)."""
    # Step 1: decay
    g_exp = ttnn.exp(g)
    state_b = ttnn.multiply(state, g_exp)
    ttnn.deallocate(g_exp)

    # Step 2: kv_mem = k_row @ state
    kv_mem = ttnn.matmul(k_row, state_b)

    # Step 3: delta = beta * (v - kv_mem)
    diff = ttnn.subtract(v, kv_mem)
    ttnn.deallocate(kv_mem)
    delta = ttnn.multiply(beta, diff)
    ttnn.deallocate(diff)

    # Step 4: state += outer(k_col, delta)
    outer = ttnn.matmul(k_col, delta)
    ttnn.deallocate(delta)
    new_state = ttnn.add(state_b, outer)
    ttnn.deallocate(state_b)
    ttnn.deallocate(outer)

    # Step 5: output = q @ new_state
    output = ttnn.matmul(q, new_state)

    # Update state in-place
    ttnn.copy(new_state, state)
    ttnn.deallocate(new_state)

    return output


# ---------------------------------------------------------------------------
# Full fused kernel (Phase A): L2 norm + gates + recurrence + RMS norm + SiLU
# ---------------------------------------------------------------------------


def _build_full_fused_device_program(
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
    num_cores,
    grid,
    state_in_l1=False,
    state_is_sharded=False,
    Nv_TP=12,
    Nk_TP=4,
    repeat_factor=3,
    key_dim_tp=512,
):
    """Build ProgramDescriptor for the full fused GDN kernel.

    Reads Q/K/V directly from conv_out via sub-tile row extraction.
    Scalar inputs (a, b) read from batched [1, B, Nv_TP] in the reader.
    Constants (neg_exp_A, dt_bias) read from [1, 1, Nv_TP] in the reader.
    z is NOT passed — handled by Python POST via ttnn.silu().
    """
    max_cores = grid.x * grid.y
    num_cores = min(num_cores, num_pairs_total, max_cores)
    pairs_per_core = num_pairs_total // num_cores
    remainder = num_pairs_total % num_cores

    core_coords = []
    for i in range(num_cores):
        core_coords.append(ttnn.CoreCoord(i % grid.x, i // grid.x))

    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in core_coords])

    # Per-core pair assignments
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    pair_offset = 0
    core_pair_counts = []

    for i, cc in enumerate(core_coords):
        n = pairs_per_core + (1 if i < remainder else 0)
        core_pair_counts.append(n)
        reader_rt_args[cc.x][cc.y] = [
            conv_out_dev.buffer_address(),  # 0
            a_dev.buffer_address(),  # 1
            b_dev.buffer_address(),  # 2
            neg_exp_A_dev.buffer_address(),  # 3
            dt_bias_dev.buffer_address(),  # 4
            norm_w_dev.buffer_address(),  # 5
            scale_dev.buffer_address(),  # 6
            rms_scale_dev.buffer_address(),  # 7
            state_dev.buffer_address(),  # 8
            rms_eps_dev.buffer_address(),  # 9
            pair_offset,  # 10
            n,  # 11
        ]
        writer_rt_args[cc.x][cc.y] = [
            output_dev.buffer_address(),
            state_dev.buffer_address(),
            pair_offset,
            n,
        ]
        pair_offset += n

    # Tile offsets for Q/K/V regions within conv_out
    key_tile_off = key_dim_tp // 32  # 16
    v_tile_off = 2 * key_tile_off  # 32

    # CB descriptors (removed cb_z = c_11, added cb_scratch = c_21)
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
        _make_cb(21, 1, core_ranges),  # cb_scratch (reader sub-tile extraction)
        _make_cb(24, 1, core_ranges),  # cb_exp_g
        _make_cb(25, Vt, core_ranges),  # cb_kv_mem
        _make_cb(26, Vt, core_ranges),  # cb_delta
        _make_cb(27, Vt, core_ranges),  # cb_delta_s
        _make_cb(28, Kt, core_ranges),  # cb_sq_acc
        _make_cb(29, 1, core_ranges),  # cb_tmp
        # cb_rec_out (c_30) removed — rec_out written directly to cb_out
        _make_cb(31, 1, core_ranges),  # cb_rms_scale (persistent)
        _make_cb(19, 1, core_ranges),  # cb_reduce_scaler (persistent)
        _make_cb(20, 1, core_ranges),  # cb_rms_eps (persistent)
    ]

    state_l1_flag = 1 if state_in_l1 else 0

    # Group cores by pair count
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

        packed_reduce_scaler = 0x3F803F80

        sharded_flag = 1 if state_is_sharded else 0

        reader_ct = [
            Kt,
            Vt,
            BF16_TILE_BYTES,
            state_l1_flag,
            packed_reduce_scaler,
            Nv_TP,
            Nk_TP,
            repeat_factor,
            key_tile_off,
            v_tile_off,
            sharded_flag,
        ]

        writer_ct = [
            Kt,
            Vt,
            BF16_TILE_BYTES,
            state_l1_flag,
            0,
            0,  # Nv_TP, out_tiles_per_batch (unused, kept for compat)
            sharded_flag,
        ]

        reader_kd = ttnn.KernelDescriptor(
            kernel_source=READER_FUSED_PATH,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=group_ranges,
            compile_time_args=reader_ct,
            runtime_args=group_reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        )

        writer_kd = ttnn.KernelDescriptor(
            kernel_source=WRITER_FUSED_PATH,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=group_ranges,
            compile_time_args=writer_ct,
            runtime_args=group_writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        )

        compute_kd = ttnn.KernelDescriptor(
            kernel_source=COMPUTE_FUSED_PATH,
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

    return ttnn.ProgramDescriptor(
        kernels=all_kernels,
        cbs=cb_descriptors,
    )


def _gdn_full_fused(
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
    num_pairs_total,
    num_cores=40,
    Nv_TP=12,
    Nk_TP=4,
    repeat_factor=3,
    key_dim_tp=512,
):
    """Execute full fused GDN kernel via registered C++ op.

    Uses ttnn.experimental.gdn_fused() which dispatches through the C++ device
    operation framework with automatic program caching and override_runtime_arguments().
    Only the 3 changing buffer addresses (conv_out, a, b) are updated per call;
    the compiled program is reused from cache.
    """
    ttnn.experimental.gdn_fused(
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
        num_pairs=num_pairs_total,
        num_cores=num_cores,
        Nv_TP=Nv_TP,
        Nk_TP=Nk_TP,
        repeat_factor=repeat_factor,
        key_dim_tp=key_dim_tp,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gdn_recurrence_fused_inplace(q, k_row, k_col, v, g, beta, state, output, num_cores=10):
    """Compute GDN recurrence and write result to output tensor.

    Tries the fused kernel first. Falls back to ttnn ops if the fused
    kernel is disabled or fails.
    """
    global _fused_available

    if _fused_available:
        try:
            logger.debug("GDN: using fused kernel")
            # Prime DRAM: touch input tensors via standard ttnn to ensure
            # writes from prior ops are visible to NOC reads in the custom kernel.
            # Workaround for Blackhole DRAM coherence with noc_async_read_tile.
            tmp = ttnn.add(q, 0.0)
            ttnn.deallocate(tmp)
            _gdn_recurrence_fused(q, k_row, k_col, v, g, beta, state, output, state, num_cores)
            return
        except Exception as e:
            logger.warning(f"Fused GDN kernel failed, falling back to ttnn ops: {e}")
            _fused_available = False

    # Fallback: ttnn ops
    logger.debug("GDN: using ttnn ops fallback")
    result = _gdn_recurrence_ttnn(q, k_row, k_col, v, g, beta, state)
    ttnn.copy(result, output)
    ttnn.deallocate(result)


def gdn_full_fused_inplace(
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
    num_cores=40,
    Nv_TP=12,
    Nk_TP=4,
    repeat_factor=3,
    key_dim_tp=512,
):
    """Full fused GDN: L2 norm + gates + recurrence.

    Reads Q/K/V directly from conv_out via sub-tile row extraction.
    Reader extracts a/b/neg_exp_A/dt_bias scalars per-pair from original shapes.
    z is NOT passed — caller handles SiLU gate via ttnn ops.

    Args:
        conv_out: [1, B, qkv_dim_tp] — post-conv1d output (batched)
        a_fused: [1, B, Nv_TP] gate input a (batched)
        b_fused: [1, B, Nv_TP] gate input b (batched)
        neg_exp_A: [1, 1, Nv_TP] precomputed -exp(A)
        dt_bias: [1, 1, Nv_TP] dt_bias constant
        norm_w: [1, 1, Dv] RMS norm weight
        scale_tt: [1, 1, 1] Q scale (Dk^-0.5)
        rms_scale_tt: [1, 1, 1] sqrt(Dv)
        rms_eps_tt: [1, 1, 1] Dv * eps
        state: [num_pairs, Dk, Dv] recurrence state
        output: [num_pairs, 1, Dv] pre-allocated output buffer
        num_pairs: total pair count (B * Nv_TP)
    """
    global _full_fused_available

    if _full_fused_available:
        try:
            logger.debug("GDN: using full fused kernel")
            _gdn_full_fused(
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
                num_pairs_total=num_pairs,
                num_cores=num_cores,
                Nv_TP=Nv_TP,
                Nk_TP=Nk_TP,
                repeat_factor=repeat_factor,
                key_dim_tp=key_dim_tp,
            )
            return
        except Exception as e:
            logger.warning(f"Full fused GDN kernel failed: {e}")
            _full_fused_available = False

    raise RuntimeError("Full fused GDN kernel not available; caller should use unfused path")
