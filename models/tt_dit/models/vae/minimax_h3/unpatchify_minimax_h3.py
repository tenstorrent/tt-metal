# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Unpatchify TILE fp32 tokens by page remap; patch_size 16."""

from __future__ import annotations

import ttnn

KERNEL_DIR = "models/tt_dit/models/vae/minimax_h3/kernels"
_CB = 0
_programs: dict = {}


def _build(
    mesh_device,
    *,
    s_pad: int,
    d: int,
    num_frames: int,
    height: int,
    out_channels: int,
    pt: int,
    p: int,
    reader_acc: tuple,
    writer_acc: tuple,
):
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
        reader_ct=ct + list(reader_acc),
        writer_ct=ct + list(writer_acc),
        cbs=[cb],
        hash=(0xC13 << 52)
        | (
            hash((s_pad, d, num_frames, height, out_channels, pt, p, tuple(ct), reader_acc, writer_acc, len(cores)))
            & ((1 << 52) - 1)
        ),
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
    """Unpatchify tiled fp32 tokens."""
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
    reader_acc = tuple(ttnn.TensorAccessorArgs(tokens).get_compile_time_args())
    writer_acc = tuple(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    key = (
        id(mesh_device),
        s_pad,
        d,
        num_frames,
        height,
        out_channels,
        patch_size_t,
        patch_size,
        reader_acc,
        writer_acc,
    )
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
            reader_acc=reader_acc,
            writer_acc=writer_acc,
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
    program.custom_program_hash = built["hash"]
    return ttnn.generic_op([tokens, out], program)
