# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""DFlash context K/V ring write as one multi-core program (Laguna; kernels/ring_write.cpp).

Rows 0..W-1 of each new [1, nkv, 32, hd] K / V tensor are copied into ring slots slots[0..W-1] of its
[1, nkv, RING, hd] ring (slots: a uint32 [1, 32] row-major device tensor, so a trace replays any slots). Replaces the
ring = ring * keep + place @ new update (a multiply, a matmul, an add and a copy per ring: ~13 us per ring)."""

from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"


def ring_write(rings, srcs, slots, written):
    """rings / srcs: equal-length lists of bf16 TILE DRAM tensors [1, nkv, RING, hd] / [1, nkv, 32, hd]."""
    assert len(rings) == len(srcs) and rings, (len(rings), len(srcs))
    device = rings[0].device()
    R = len(rings)
    nkv, ring_len, hd = rings[0].shape[1], rings[0].shape[2], rings[0].shape[3]
    assert ring_len % 32 == 0 and hd % 32 == 0 and 1 <= int(written) <= 32, (ring_len, hd, written)
    for t in list(rings) + list(srcs):
        assert t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT and not t.is_sharded(), (t.dtype, t.layout)
    units = R * nkv * (hd // 32)
    gs = device.compute_with_storage_grid_size()
    assert units <= gs.x * gs.y, units
    grid = ttnn.num_cores_to_corerangeset(units, gs, True)
    k = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "ring_write.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[gs.x, R, nkv, hd // 32, ring_len // 32]
        + list(ttnn.TensorAccessorArgs(slots).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(srcs[0]).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(rings[0]).get_compile_time_args()),
        common_runtime_args=[slots.buffer_address(), int(written)]
        + [t.buffer_address() for t in srcs]
        + [t.buffer_address() for t in rings],
        config=ttnn.ReaderConfigDescriptor(),
    )

    def cb(i, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)],
        )

    ttnn.generic_op([slots] + list(srcs) + list(rings), ttnn.ProgramDescriptor(kernels=[k], semaphores=[],
                                                                            cbs=[cb(0, 2048, 2), cb(1, 128, 1)]))  # fmt: skip


def ring_write_batched(rings, srcs, dests, rows_per_user=1):
    """Batched rings: rings [U, nkv, RING, hd], srcs [1, nkv, T, hd] or [T / 32, nkv, 32, hd] (T a multiple of 32, at
    most 512), both bf16 TILE DRAM; dests: uint32 [1, T] row-major device tensor, dests[r] = user * RING + slot for
    row r (0xFFFFFFFF: skip). One core per (ring, kv head, head_dim tile); see kernels/ring_write_batched.cpp."""
    assert len(rings) == len(srcs) and rings, (len(rings), len(srcs))
    device = rings[0].device()
    R = len(rings)
    nkv, ring_len, hd = rings[0].shape[1], rings[0].shape[2], rings[0].shape[3]
    src_batched = int(srcs[0].shape[0]) > 1
    T = int(srcs[0].shape[0]) * int(srcs[0].shape[2])
    assert ring_len % 32 == 0 and hd % 32 == 0 and T % 32 == 0 and T <= 512, (ring_len, hd, T)
    assert not src_batched or int(srcs[0].shape[2]) == 32, srcs[0].shape
    for t in list(rings) + list(srcs):
        assert t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT and not t.is_sharded(), (t.dtype, t.layout)
    units = R * nkv * (hd // 32)
    gs = device.compute_with_storage_grid_size()
    assert units <= gs.x * gs.y, units
    # row groups (whole users each, so no two cores write one ring tile) over the rest of the grid
    users = T // int(rows_per_user)
    groups = max(1, min(users, (gs.x * gs.y) // units))
    while users % groups:
        groups -= 1
    group_rows = T // groups
    grid = ttnn.num_cores_to_corerangeset(units * groups, gs, True)
    k = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "ring_write_batched.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[gs.x, R, nkv, hd // 32, ring_len, T, int(src_batched), group_rows]
        + list(ttnn.TensorAccessorArgs(dests).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(srcs[0]).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(rings[0]).get_compile_time_args()),
        common_runtime_args=[dests.buffer_address(), T]
        + [t.buffer_address() for t in srcs]
        + [t.buffer_address() for t in rings],
        config=ttnn.ReaderConfigDescriptor(),
    )

    def cb(i, page, pages, fmt):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page)],
        )

    ttnn.generic_op([dests] + list(srcs) + list(rings), ttnn.ProgramDescriptor(
        kernels=[k], semaphores=[], cbs=[cb(0, 2048, -(-group_rows // 32) + 2, ttnn.bfloat16),
                                         cb(1, T * 4, 1, ttnn.uint32)]))  # fmt: skip
