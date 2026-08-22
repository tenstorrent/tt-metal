# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused GDN decode ops (generic_op + JIT kernels).

Two ops replace the ~55-op composite GDN decode chain between the qkvzab in-projection
and the out-projection:

  1. conv_shift_silu:  conv = silu(FIR(qkvzab.qkv, conv shift register)) with the
     shift-register writeback done in-kernel (states updated in place).
  2. recurrence:       head prep (GQA, sigmoid/softplus gates), L2 norms, the fp32
     delta-rule state update (in place), and the gated rmsnorm + silu(z) output gate.

State tensors are updated in place, so both ops are decode-trace compatible (all
addresses persist across replays). Outputs must be preallocated by the caller
(generic_op contract: last io_tensor is the output).
"""

import struct

import ttnn

_KDIR = "models/demos/blackhole/qwen36/tt/gdn/fused_decode/kernels"

_TILE = 32
_BF16_TILE = 2048
_FP32_TILE = 4096


def _fbits(x):
    """fp32 bit pattern as uint32 (kernels consume scalars via binop_with_scalar)."""
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _is_dram(t):
    return 1 if t.memory_config().buffer_type == ttnn.BufferType.DRAM else 0


def _cb(index, data_format, page_size, num_pages, core_ranges):
    fmt = ttnn.CBFormatDescriptor(
        buffer_index=index,
        data_format=data_format,
        page_size=page_size,
        tile=ttnn.TileDescriptor(_TILE, _TILE, False),
    )
    return ttnn.CBDescriptor(total_size=num_pages * page_size, core_ranges=core_ranges, format_descriptors=[fmt])


def _reader_config():
    return ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default)


def _writer_config():
    return ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_1_default)


def _runtime_args(cores, per_core):
    ra = ttnn.RuntimeArgs()
    for core, args in zip(cores, per_core):
        ra[core.x][core.y] = [int(a) for a in args]
    return ra


def conv_shift_silu(qkvzab, conv_states, conv_taps, conv_out):
    """FIR conv + SiLU over the decode shift register, with in-place shift writeback.

    qkvzab:      [1, B, qkvzab_dim] bf16 TILE (qkv = leading columns).
    conv_states: list of K=4 persistent [1, B, qkv_dim] bf16 TILE tensors; updated in
                 place (st0<-st1, st1<-st2, st2<-st3, st3<-qkv).
    conv_taps:   list of K=4 [1, 1, qkv_dim] bf16 TILE tensors.
    conv_out:    preallocated [1, B, qkv_dim] bf16 TILE output.
    """
    assert len(conv_states) == 4 and len(conv_taps) == 4, "decode conv is a fixed 4-tap FIR"
    for t in [qkvzab, conv_out, *conv_states, *conv_taps]:
        assert t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT, "conv operands must be bf16 TILE"
    assert qkvzab.shape[-2] <= _TILE, "decode conv expects a single tile row (B <= 32)"
    qkv_dim = conv_out.shape[-1]
    assert qkv_dim % _TILE == 0
    wt = qkv_dim // _TILE

    device = qkvzab.device()
    grid = device.compute_with_storage_grid_size()
    ncores = min(wt, grid.x * grid.y)
    core_ranges = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    cores = ttnn.corerange_to_cores(core_ranges, row_wise=True)

    # Contiguous wi ranges, ceil/floor split.
    base, rem = wt // ncores, wt % ncores
    spans = []
    start = 0
    for i in range(ncores):
        cnt = base + (1 if i < rem else 0)
        spans.append((start, cnt))
        start += cnt

    named_reader = [
        ("qkvzab_is_dram", _is_dram(qkvzab)),
        ("st_is_dram", _is_dram(conv_states[1])),
        ("tap_is_dram", _is_dram(conv_taps[0])),
    ]
    named_writer = [
        ("conv_is_dram", _is_dram(conv_out)),
        ("st_is_dram", _is_dram(conv_states[0])),
    ]

    reader = ttnn.KernelDescriptor(
        kernel_source=f"{_KDIR}/gdn_conv_shift_silu_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=[],
        named_compile_time_args=named_reader,
        common_runtime_args=[
            qkvzab.buffer_address(),
            conv_states[1].buffer_address(),
            conv_states[2].buffer_address(),
            conv_states[3].buffer_address(),
            conv_taps[0].buffer_address(),
            conv_taps[1].buffer_address(),
            conv_taps[2].buffer_address(),
            conv_taps[3].buffer_address(),
        ],
        runtime_args=_runtime_args(cores, [(s, c) for (s, c) in spans]),
        config=_reader_config(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{_KDIR}/gdn_conv_shift_silu_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=[],
        named_compile_time_args=named_writer,
        common_runtime_args=[
            conv_out.buffer_address(),
            conv_states[0].buffer_address(),
            conv_states[1].buffer_address(),
            conv_states[2].buffer_address(),
            conv_states[3].buffer_address(),
        ],
        runtime_args=_runtime_args(cores, [(s, c) for (s, c) in spans]),
        config=_writer_config(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{_KDIR}/gdn_conv_shift_silu_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=[],
        named_compile_time_args=[],
        common_runtime_args=[],
        runtime_args=_runtime_args(cores, [(c,) for (_, c) in spans]),
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
        ),
    )

    cbs = [
        _cb(0, ttnn.bfloat16, _BF16_TILE, 8, core_ranges),  # cb_x: [st1,st2,st3,qkv], double-buffered
        _cb(1, ttnn.bfloat16, _BF16_TILE, 8, core_ranges),  # cb_taps
        _cb(2, ttnn.bfloat16, _BF16_TILE, 8, core_ranges),  # cb_shift (writer copy)
        _cb(3, ttnn.bfloat16, _BF16_TILE, 2, core_ranges),  # cb_out
    ]

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs)
    io_tensors = [qkvzab] + list(conv_states) + list(conv_taps) + [conv_out]
    return ttnn.generic_op(io_tensors, program)


def recurrence(conv_out, qkvzab, rec_state, dt_bias, neg_exp_a, norm_w, out, *, nk, nv, dk, dv, b_rows, eps=1e-6):
    """Fused GDN decode recurrence: head prep + L2 norms + delta rule + gated out norm.

    conv_out:  [1, B, qkv_dim] bf16 TILE (silu'd conv output; q|k|v blocks).
    qkvzab:    [1, B, qkvzab_dim] bf16 TILE (z at qkv_dim, a/b in the tile at qkvz_dim).
    rec_state: [B, Nv, Dk, Dv] fp32 TILE, updated IN PLACE.
    dt_bias, neg_exp_a: [1, 1, Nv] bf16 TILE per-head gate params.
    norm_w:    [1, 1, Dv] bf16 TILE gated-norm weight (no +1).
    out:       preallocated [1, B, Nv*Dv] fp32 TILE gated output.
    b_rows:    active batch rows (rows [b_rows, 32) of out are zeroed).
    """
    assert rec_state.dtype == ttnn.float32, "fused GDN decode requires the fp32 recurrent state"
    assert out.dtype == ttnn.float32, "gated output is fp32 (matches the composite path)"
    for t in [conv_out, qkvzab, dt_bias, neg_exp_a, norm_w]:
        assert t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT, "recurrence bf16 operands must be bf16 TILE"
    assert dk % _TILE == 0 and dv % _TILE == 0
    # a and beta must share one 32-wide tile (a at column vh, beta at NV+vh).
    assert 2 * nv <= _TILE, f"gate logits span two tiles at nv={nv}; the reader assumes one"
    dkt, dvt = dk // _TILE, dv // _TILE
    rf = nv // nk
    qkv_dim = conv_out.shape[-1]
    kd_t = nk * dkt  # k block tile offset (q block spans [0, kd_t))
    voff_t = 2 * nk * dkt
    zoff_t = qkv_dim // _TILE
    # a/b live in the first tile past the qkvz block; qkvz_dim = qkv_dim + z_dim (z_dim = nv*dv).
    ab_t = (qkv_dim + nv * dv) // _TILE

    device = conv_out.device()
    grid = device.compute_with_storage_grid_size()
    units = b_rows * nv
    assert units <= grid.x * grid.y, f"one (b, head) per core: {units} units > {grid.x * grid.y} cores"
    core_ranges = ttnn.num_cores_to_corerangeset(units, grid, row_wise=True)
    cores = ttnn.corerange_to_cores(core_ranges, row_wise=True)
    unit_args = [(u // nv, u % nv) for u in range(units)]  # (b, vh)

    named_reader = [
        ("nv", nv),
        ("rf", rf),
        ("dkt", dkt),
        ("dvt", dvt),
        ("kd_t", kd_t),
        ("voff_t", voff_t),
        ("zoff_t", zoff_t),
        ("ab_t", ab_t),
        ("conv_is_dram", _is_dram(conv_out)),
        ("qkvzab_is_dram", _is_dram(qkvzab)),
        ("state_is_dram", _is_dram(rec_state)),
        ("params_is_dram", _is_dram(dt_bias)),
    ]
    named_compute = [
        ("nv", nv),
        ("dkt", dkt),
        ("dvt", dvt),
        ("eps_bits", _fbits(eps)),
        ("scale_bits", _fbits(dk**-0.5)),
        ("sp_beta_bits", _fbits(1.0)),
        ("sp_beta_recip_bits", _fbits(1.0)),
        ("sp_thr_bits", _fbits(20.0)),
        ("inv_dv_bits", _fbits(1.0 / dv)),
        ("norm_eps_bits", _fbits(eps)),
        ("seq_rows", 0),
    ]
    named_writer = [
        ("nv", nv),
        ("dkt", dkt),
        ("dvt", dvt),
        ("b_rows", b_rows),
        ("seq_rows", 0),
        ("state_is_dram", _is_dram(rec_state)),
        ("out_is_dram", _is_dram(out)),
    ]

    reader = ttnn.KernelDescriptor(
        kernel_source=f"{_KDIR}/gdn_recurrence_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=[],
        named_compile_time_args=named_reader,
        common_runtime_args=[
            conv_out.buffer_address(),
            qkvzab.buffer_address(),
            rec_state.buffer_address(),
            dt_bias.buffer_address(),
            neg_exp_a.buffer_address(),
            norm_w.buffer_address(),
        ],
        runtime_args=_runtime_args(cores, unit_args),
        config=_reader_config(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{_KDIR}/gdn_recurrence_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=[],
        named_compile_time_args=named_writer,
        common_runtime_args=[rec_state.buffer_address(), out.buffer_address()],
        runtime_args=_runtime_args(cores, unit_args),
        config=_writer_config(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{_KDIR}/gdn_recurrence_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=[],
        named_compile_time_args=named_compute,
        common_runtime_args=[],
        runtime_args=_runtime_args(cores, unit_args),
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
        ),
    )

    bf, fp = ttnn.bfloat16, ttnn.float32
    kv = dkt * dvt
    cbs = [
        _cb(0, bf, _BF16_TILE, dkt, core_ranges),  # qin
        _cb(1, bf, _BF16_TILE, dkt, core_ranges),  # kin
        _cb(2, bf, _BF16_TILE, dvt, core_ranges),  # vin
        _cb(3, bf, _BF16_TILE, dvt, core_ranges),  # zin
        _cb(4, bf, _BF16_TILE, 1, core_ranges),  # ab
        _cb(5, bf, _BF16_TILE, 1, core_ranges),  # dt_bias
        _cb(6, bf, _BF16_TILE, 1, core_ranges),  # neg_exp_A
        _cb(7, bf, _BF16_TILE, dvt, core_ranges),  # norm_w
        _cb(8, fp, _FP32_TILE, 1, core_ranges),  # ones (fp32: same-format with the intermediates)
        _cb(9, fp, _FP32_TILE, kv, core_ranges),  # h
        _cb(10, fp, _FP32_TILE, 1, core_ranges),  # beta_full
        _cb(11, fp, _FP32_TILE, 1, core_ranges),  # decay_full
        _cb(12, fp, _FP32_TILE, max(4, dkt), core_ranges),  # scr
        _cb(13, fp, _FP32_TILE, 1, core_ranges),  # decay_s
        _cb(14, fp, _FP32_TILE, 1, core_ranges),  # beta_s
        _cb(15, fp, _FP32_TILE, max(dkt, dvt), core_ranges),  # sq
        _cb(16, fp, _FP32_TILE, 1, core_ranges),  # colscale
        _cb(17, fp, _FP32_TILE, dkt, core_ranges),  # qn
        _cb(18, fp, _FP32_TILE, dkt, core_ranges),  # kn
        _cb(19, fp, _FP32_TILE, kv, core_ranges),  # hd
        _cb(20, fp, _FP32_TILE, dvt, core_ranges),  # vread / silu(z)
        _cb(21, fp, _FP32_TILE, dvt, core_ranges),  # delta / normed o
        _cb(22, fp, _FP32_TILE, dvt, core_ranges),  # delta*beta / normed*w
        _cb(23, fp, _FP32_TILE, dkt, core_ranges),  # k col-broadcast
        _cb(24, fp, _FP32_TILE, kv, core_ranges),  # outer
        _cb(25, fp, _FP32_TILE, kv, core_ranges),  # h_new (writer)
        _cb(26, fp, _FP32_TILE, dvt, core_ranges),  # o
        _cb(27, fp, _FP32_TILE, dvt, core_ranges),  # out
        _cb(28, fp, _FP32_TILE, 1, core_ranges),  # writer zero scratch
    ]

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs)
    io_tensors = [conv_out, qkvzab, rec_state, dt_bias, neg_exp_a, norm_w, out]
    return ttnn.generic_op(io_tensors, program)


def recurrence_seq_rows(
    conv_out, qkvzab, rec_state, state_stash, dt_bias, neg_exp_a, norm_w, out, *, nk, nv, dk, dv, users, w, eps=1e-6
):
    """Speculative-verify GDN recurrence: W candidate rows per user, sequential
    IN-KERNEL, with per-row state stashes and NO writeback of the anchor state.

    Row u*w + t of the width-(users*w) activations is user u's candidate t at its
    own position. One core per (user, v-head) loads rec_state[u, vh] (the user's
    committed ANCHOR) into L1 once, loops the w rows locally, and writes each
    step's post-update state to state_stash[u, t, vh] — the host later commits by
    row-copying state_stash[u, m_u] into rec_state[u] (commit-by-select).

    conv_out:    [1, users*w, qkv_dim] bf16 TILE (users*w <= 32: one tile row).
    qkvzab:      [1, users*w, qkvzab_dim] bf16 TILE.
    rec_state:   [users, Nv, Dk, Dv] fp32 TILE — READ ONLY (never written).
    state_stash: [users, w, Nv, Dk, Dv] fp32 TILE, fully overwritten.
    out:         preallocated [1, users*w, Nv*Dv] fp32 TILE.
    """
    assert rec_state.dtype == ttnn.float32 and state_stash.dtype == ttnn.float32
    assert out.dtype == ttnn.float32
    for t in [conv_out, qkvzab, dt_bias, neg_exp_a, norm_w]:
        assert t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT
    assert dk % _TILE == 0 and dv % _TILE == 0
    assert 2 * nv <= _TILE
    assert w >= 1 and users * w <= _TILE, f"users*w rows must fit one tile row (got {users}x{w})"
    assert list(state_stash.shape) == [users, w, nv, dk, dv], f"stash shape {list(state_stash.shape)}"
    dkt, dvt = dk // _TILE, dv // _TILE
    rf = nv // nk
    qkv_dim = conv_out.shape[-1]
    kd_t = nk * dkt
    voff_t = 2 * nk * dkt
    zoff_t = qkv_dim // _TILE
    ab_t = (qkv_dim + nv * dv) // _TILE
    b_rows = users * w

    device = conv_out.device()
    grid = device.compute_with_storage_grid_size()
    units = users * nv
    assert units <= grid.x * grid.y, f"one (user, head) per core: {units} units > {grid.x * grid.y} cores"
    core_ranges = ttnn.num_cores_to_corerangeset(units, grid, row_wise=True)
    cores = ttnn.corerange_to_cores(core_ranges, row_wise=True)
    unit_args = [(uidx // nv, uidx % nv) for uidx in range(units)]  # (user, vh)

    named_reader = [
        ("nv", nv),
        ("rf", rf),
        ("dkt", dkt),
        ("dvt", dvt),
        ("kd_t", kd_t),
        ("voff_t", voff_t),
        ("zoff_t", zoff_t),
        ("ab_t", ab_t),
        ("conv_is_dram", _is_dram(conv_out)),
        ("qkvzab_is_dram", _is_dram(qkvzab)),
        ("state_is_dram", _is_dram(rec_state)),
        ("params_is_dram", _is_dram(dt_bias)),
    ]
    named_compute = [
        ("nv", nv),
        ("dkt", dkt),
        ("dvt", dvt),
        ("eps_bits", _fbits(eps)),
        ("scale_bits", _fbits(dk**-0.5)),
        ("sp_beta_bits", _fbits(1.0)),
        ("sp_beta_recip_bits", _fbits(1.0)),
        ("sp_thr_bits", _fbits(20.0)),
        ("inv_dv_bits", _fbits(1.0 / dv)),
        ("norm_eps_bits", _fbits(eps)),
        ("seq_rows", w),
    ]
    named_writer = [
        ("nv", nv),
        ("dkt", dkt),
        ("dvt", dvt),
        ("b_rows", b_rows),
        ("seq_rows", w),
        # The writer's state accessor targets the STASH in seq mode.
        ("state_is_dram", _is_dram(state_stash)),
        ("out_is_dram", _is_dram(out)),
    ]

    reader = ttnn.KernelDescriptor(
        kernel_source=f"{_KDIR}/gdn_recurrence_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=[],
        named_compile_time_args=named_reader,
        common_runtime_args=[
            conv_out.buffer_address(),
            qkvzab.buffer_address(),
            rec_state.buffer_address(),
            dt_bias.buffer_address(),
            neg_exp_a.buffer_address(),
            norm_w.buffer_address(),
        ],
        runtime_args=_runtime_args(cores, unit_args),
        config=_reader_config(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{_KDIR}/gdn_recurrence_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=[],
        named_compile_time_args=named_writer,
        common_runtime_args=[state_stash.buffer_address(), out.buffer_address()],
        runtime_args=_runtime_args(cores, unit_args),
        config=_writer_config(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{_KDIR}/gdn_recurrence_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_ranges,
        compile_time_args=[],
        named_compile_time_args=named_compute,
        common_runtime_args=[],
        runtime_args=_runtime_args(cores, unit_args),
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
        ),
    )

    bf, fp = ttnn.bfloat16, ttnn.float32
    kv = dkt * dvt
    cbs = [
        _cb(0, bf, _BF16_TILE, dkt, core_ranges),  # qin
        _cb(1, bf, _BF16_TILE, dkt, core_ranges),  # kin
        _cb(2, bf, _BF16_TILE, dvt, core_ranges),  # vin (retained across the w steps)
        _cb(3, bf, _BF16_TILE, dvt, core_ranges),  # zin (retained)
        _cb(4, bf, _BF16_TILE, 1, core_ranges),  # ab
        _cb(5, bf, _BF16_TILE, 1, core_ranges),  # dt_bias
        _cb(6, bf, _BF16_TILE, 1, core_ranges),  # neg_exp_A
        _cb(7, bf, _BF16_TILE, dvt, core_ranges),  # norm_w (retained)
        _cb(8, fp, _FP32_TILE, 1, core_ranges),  # ones
        _cb(9, fp, _FP32_TILE, kv, core_ranges),  # h (anchor)
        _cb(10, fp, _FP32_TILE, 1, core_ranges),  # beta_full (retained)
        _cb(11, fp, _FP32_TILE, 1, core_ranges),  # decay_full (retained)
        _cb(12, fp, _FP32_TILE, max(4, dkt), core_ranges),  # scr
        _cb(13, fp, _FP32_TILE, 1, core_ranges),  # decay_s
        _cb(14, fp, _FP32_TILE, 1, core_ranges),  # beta_s
        _cb(15, fp, _FP32_TILE, max(dkt, dvt), core_ranges),  # sq
        _cb(16, fp, _FP32_TILE, 1, core_ranges),  # colscale
        _cb(17, fp, _FP32_TILE, dkt, core_ranges),  # qn (retained)
        _cb(18, fp, _FP32_TILE, dkt, core_ranges),  # kn (retained)
        _cb(19, fp, _FP32_TILE, kv, core_ranges),  # hd
        _cb(20, fp, _FP32_TILE, dvt, core_ranges),  # vread / silu(z)
        _cb(21, fp, _FP32_TILE, dvt, core_ranges),  # delta / normed o
        _cb(22, fp, _FP32_TILE, dvt, core_ranges),  # delta*beta / normed*w
        _cb(23, fp, _FP32_TILE, dkt, core_ranges),  # k col-broadcast
        _cb(24, fp, _FP32_TILE, kv, core_ranges),  # outer
        _cb(25, fp, _FP32_TILE, kv, core_ranges),  # h_new (next step's input)
        _cb(26, fp, _FP32_TILE, dvt, core_ranges),  # o
        _cb(27, fp, _FP32_TILE, dvt, core_ranges),  # out
        _cb(28, fp, _FP32_TILE, 1, core_ranges),  # writer zero scratch
        _cb(29, fp, _FP32_TILE, kv, core_ranges),  # h stash stream (writer)
    ]

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs)
    io_tensors = [conv_out, qkvzab, rec_state, state_stash, dt_bias, neg_exp_a, norm_w, out]
    return ttnn.generic_op(io_tensors, program)
