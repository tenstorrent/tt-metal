# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GDN conv + SiLU as one device op (milestone 1), via ttnn.generic_op.

The kernel body that then gets lifted into fused_recurrent_gated_delta_rule. Built as a generic_op
first so each iteration is a kernel-source edit rather than a C++ rebuild plus install.

Scope note, corrected: the conv output is written out as its own tensor, so q/k can still be
L2-normalised by the caller exactly as today. Folding the conv in does NOT force the norm inside.
That keeps this milestone small, and the norm (ms 2) stays a separate, independently gated step.

Contract (bf16 TILE, interleaved, one Tensix core per value head):
    qkv    [1, 1, C]   raw pre-conv projection output, C = 2*key_dim + value_dim
    state  [K, 1, C]   conv shift register, row j the (t-K+j)-th input, oldest first
    taps   [K, 1, C]   the K conv taps
  ->
    conv   [1, 1, C]   conv + SiLU for this head's channels, other columns untouched
    shift  [K, 1, C]   the ring after this token: row j <- j+1, row K-1 <- qkv

The op writes only the columns it owns, so `conv` and `shift` can be the caller's full-width buffers
copied once, or fresh allocations. Here they are fresh, so the test can compare whole tensors.
"""

from __future__ import annotations

import torch
import ttnn

KERNEL_ROOT = "models/demos/blackhole/bonsai2/tp1/ops/conv_silu"
TILE = 32


def _cb(index: int, pages: int, dtype, cores) -> ttnn.CBDescriptor:
    page = {ttnn.bfloat16: 2048, ttnn.float32: 4096}[dtype]
    return ttnn.CBDescriptor(
        total_size=pages * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def _acc(t: ttnn.Tensor):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def geometry(key_dim: int, value_dim: int, num_value_heads: int, gqa: int, K: int) -> dict:
    """Tile geometry, all derived from the model's real head counts."""
    nk = num_value_heads // gqa
    assert nk * gqa == num_value_heads, f"{num_value_heads} value heads not divisible by gqa {gqa}"
    kd_t = (key_dim // nk) // TILE
    v_t = (value_dim // num_value_heads) // TILE
    assert key_dim == nk * kd_t * TILE, f"key_dim {key_dim} != {nk} heads x {kd_t} x {TILE}"
    assert value_dim == num_value_heads * v_t * TILE, f"value_dim {value_dim} != {num_value_heads} x {v_t} x {TILE}"
    C = 2 * key_dim + value_dim
    return {
        "K": int(K),
        "kd_t": kd_t,
        "v_t": v_t,
        "c_t": C // TILE,
        "C": C,
        "nv": num_value_heads,
        "nk": nk,
        "per_core": 2 * kd_t + v_t,
    }


def tap_diagonals(taps: torch.Tensor) -> torch.Tensor:
    """[K,1,C] taps -> [K,1,C] diagonal tiles, so the kernel's conv is (state @ diag(tap)).

    Each channel becomes a 32x32 diagonal tile in the [1,1,C] tile grid: element [r][c] of the tile is
    tap[c] iff r == c. Then (state_tile @ diag_tile)[r][c] = state[r][c] * tap[c], which is the
    elementwise product the conv needs -- and unlike mul_tiles it ACCUMULATES when repeated, because
    it is a matmul. Built once per layer (the taps are static), not per token.
    """
    K, _, C = taps.shape
    assert C % TILE == 0, f"conv width {C} must be a multiple of {TILE}"
    # A [1,1,C] TILE tensor holds C/32 tiles; tile t carries channels [32t, 32t+32) in its row 0 only.
    t3 = taps.reshape(K, C // TILE, TILE)  # (K, tiles, 32 live channels)
    out = torch.zeros(K, C // TILE, TILE, TILE, dtype=taps.dtype)
    idx = torch.arange(TILE)
    out[:, :, idx, idx] = t3  # tile t: diagonal r==c <- channel r
    # [K, C/32, 32, 32]: one real 32x32 tile per channel, which is what the kernel needs. The
    # [K,1,C] form cannot hold it (it gives 32 elements per tile, not 1024).
    return out.reshape(K, C // TILE, TILE, TILE)


def conv_silu(qkv, state, taps, key_dim: int, value_dim: int, num_value_heads: int, gqa: int):
    """See the module docstring. Returns (conv, shift). `taps` may be raw taps or diagonals."""
    g = geometry(int(key_dim), int(value_dim), int(num_value_heads), int(gqa), int(state.shape[0]))
    device = qkv.device()
    nv, K, C = g["nv"], g["K"], g["C"]
    # Place the nv per-head cores on the WORKER grid, not at (0,0): kernels cannot live on dispatch
    # cores. The compute grid is 11x10 on this part, so lay the heads out row-major over it.
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    assert nv <= gx * gy, f"{nv} value heads exceed {gx*gy} worker cores"
    head_cores = [ttnn.CoreCoord(h % gx, h // gx) for h in range(nv)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in head_cores])

    def alloc(shape):
        # from_torch of zeros guarantees a zeroed buffer; allocate_tensor_on_device does not.
        return ttnn.from_torch(
            torch.zeros(*shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    conv, shift = alloc([1, 1, C]), alloc([K, 1, C])
    # allocate_tensor_on_device does NOT zero, so every column a core does not own holds whatever was
    # in that allocation. Zero both outputs first so a stray write (or a missing write) shows up as a
    # deviation from zero rather than as garbage that can be inf.
    # allocate_tensor_on_device does NOT zero the buffer, so every column a core does not own holds
    # whatever was in that allocation (which is where the inf and the wrong in-slice values come from:
    # an unowned tile can be read as a diagonal). Zero both outputs before the op so a stray write or a
    # missing write shows up as a deviation from zero.

    c_t, per_core = g["c_t"], g["per_core"]
    cbs = [
        _cb(0, per_core, ttnn.bfloat16, cores),      # the new input row, this head's channels
        _cb(1, K * per_core, ttnn.bfloat16, cores),  # K state rows, same channels
        _cb(2, K * per_core, ttnn.bfloat16, cores),  # K diagonal-tap rows, same channels
        _cb(3, 2 * per_core, ttnn.bfloat16, cores),  # conv+silu output, double buffered
        _cb(4, K * per_core, ttnn.bfloat16, cores),  # the shifted ring, for the writer only
    ]

    coord_to_head = {(c.x, c.y): h for h, c in enumerate(head_cores)}

    def h_of(coord):
        return coord_to_head[coord]

    def rt(core):
        return (
            (core.x, core.y),
            [
                qkv.buffer_address(),
                state.buffer_address(),
                taps.buffer_address(),
                conv.buffer_address(),
                shift.buffer_address(),
                nv,
                int(gqa),
                g["kd_t"],
                g["v_t"],
                h_of((core.x, core.y)),  # this core's value head; last so both kernels read it at index 9
            ],
        )

    core_list = head_cores
    ct = [g["kd_t"], g["v_t"], c_t, K, per_core]
    rt_args = [rt(c) for c in core_list]

    reader = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_ROOT}/reader_conv_silu.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[*ct, *_acc(qkv), *_acc(state), *_acc(taps)],
        runtime_args=list(rt_args),
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_ROOT}/compute_conv_silu.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[g["kd_t"], g["v_t"], K, per_core],
        runtime_args=list(rt_args),
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_ROOT}/writer_conv_silu.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[*ct, *_acc(conv), *_acc(shift)],
        runtime_args=[
            ((c.x, c.y), [conv.buffer_address(), shift.buffer_address(), nv, int(gqa), g["kd_t"], g["v_t"], h])
            for h, c in enumerate(core_list)
        ],
        config=ttnn.WriterConfigDescriptor(),
    )

    ttnn.generic_op(
        [qkv, state, taps, conv, shift],
        ttnn.ProgramDescriptor(kernels=[reader, compute, writer], cbs=cbs),
    )
    return conv, shift
