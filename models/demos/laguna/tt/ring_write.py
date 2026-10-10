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
