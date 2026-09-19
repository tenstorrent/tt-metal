# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Unpatchify as one data-movement program over the TILE-layout fp32 token tensor.

``unpatchify_device`` (stitch_device_minimax_h3.py) is untilize -> slice -> reshape copy -> rank-8 permute -> reshape
copy, ~3.4 ms per decoder wave per device. On the tiled tensor the same permutation is a page remap: the 16 tokens of
a (t, h) unit are one half of a tile row, and the 1 KB face holding their 16-feature group ``(c, f, yy)`` is exactly
canvas row ``(c, t*pt + f, h*p + yy)``. Two kernels (``kernels/unpatchify_{reader,writer}.cpp``) stage each unit's
96 half-tiles and write its 192 rows; bit-identical to the reference by construction (no arithmetic).
Requires ``patch_size == 16`` and a 16-patch-wide tile (the face geometry); fp32 tokens.
"""

from __future__ import annotations

import ttnn

KERNEL_DIR = "models/tt_dit/models/vae/minimax_h3/kernels"
_CB = 0
_TILE = 4096
_programs: dict = {}


def _build(mesh_device, *, s_pad: int, d: int, num_frames: int, height: int, out_channels: int, pt: int, p: int, x, out):
    grid = mesh_device.compute_with_storage_grid_size()
    cores = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)]
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    units = num_frames * height
    base, rem = divmod(units, len(cores))
    rt = ttnn.RuntimeArgs()
    u = 0
    for j, (cx, cy) in enumerate(cores):
        n = base + (1 if j < rem else 0)
        rt[cx][cy] = [u, u + n]
        u += n
    assert u == units
    d_tiles = d // 32
    ct = [_CB, d_tiles, height, num_frames, out_channels, pt, p]
    unit_bytes = d_tiles * 2048
    cb = ttnn.CBDescriptor(
        total_size=2 * unit_bytes,
        core_ranges=core_grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=_CB, data_format=ttnn.float32, page_size=unit_bytes)],
    )
    return dict(
        core_grid=core_grid,
        rt=rt,
        reader_ct=ct + ttnn.TensorAccessorArgs(x).get_compile_time_args(),
        writer_ct=ct + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
        cbs=[cb],
        # Deterministic across processes (ints only, no str hashing); every compile-time and work-split input is in it,
        # since generic_op trusts the hash on a program-cache hit.
        hash=(0xC13 << 52) | (hash((s_pad, d, num_frames, height, out_channels, pt, p, tuple(ct), len(cores))) & ((1 << 52) - 1)),
    )


def unpatchify_tiled(
    tokens: ttnn.Tensor,
    *,
    num_frames: int,
    height: int,
    width: int,
    out_channels: int = 3,
    patch_size: int = 16,
    patch_size_t: int = 4,
) -> ttnn.Tensor:
    """``(1, S_pad, C*pt*p*p)`` TILE fp32 tokens (rows >= T*H*W ignored) to ``(1, C, T*pt, H*p, W*p)`` fp32 ROW_MAJOR."""
    assert tokens.layout == ttnn.TILE_LAYOUT and tokens.dtype == ttnn.float32, "tiled fp32 tokens only"
    batch, s_pad, d = (int(v) for v in tokens.shape)
    assert batch == 1, "one tile at a time"
    assert width == 16 and patch_size == 16, "the face remap needs 16-wide patches in a 16-patch-wide tile"
    assert d == out_channels * patch_size_t * patch_size * patch_size and d % 32 == 0
    assert num_frames * height * width <= s_pad
    mesh_device = tokens.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, out_channels, num_frames * patch_size_t, height * patch_size, width * patch_size]),
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh_device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    key = (id(mesh_device), s_pad, d, num_frames, height, out_channels, patch_size_t, patch_size)
    built = _programs.get(key)
    if built is None:
        built = _build(
            mesh_device,
            s_pad=s_pad,
            d=d,
            num_frames=num_frames,
            height=height,
            out_channels=out_channels,
            pt=patch_size_t,
            p=patch_size,
            x=tokens,
            out=out,
        )
        _programs[key] = built
    common = dict(core_ranges=built["core_grid"], runtime_args=built["rt"])
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/unpatchify_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        compile_time_args=built["reader_ct"],
        common_runtime_args=[tokens.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
        **common,
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/unpatchify_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        compile_time_args=built["writer_ct"],
        common_runtime_args=[out.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
        **common,
    )
    program = ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=built["cbs"])
    try:
        program.custom_program_hash = built["hash"]
    except (AttributeError, TypeError):
        pass
    return ttnn.generic_op([tokens, out], program)
