# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Experimental single-token convolution with compact users and in-place history.

One tile row carries users, rather than four time rows per user. The arithmetic
and BF16 partial-rounding boundaries match the existing causal-convolution op.
Outputs are caller-owned compact [1,B,C] tiles for direct GDN preparation.
No serving or model policy selects this experiment yet.
"""

from pathlib import Path

from models.demos.qwen38_27b_qb2.tt.gdn_step.op import work_items

HERE = Path(__file__).parent
WIDTHS = (512, 512, 1536)


def convolution(qkv, history, taps, outputs, *, compact_input=False):
    """Write compact Q/K/V and shift each user's three-row history in place.

    Input is either [B,1,W] or compact [1,B,W], where W can include the packed
    projection's z/gate tail. Only the first 2560 channels are consumed. Distinct
    channel workers own disjoint history ranges and read all four taps before
    writing history. Caller-owned outputs must survive every trace replay.
    """
    import ttnn

    if type(compact_input) is not bool:
        raise ValueError("compact_input must be a Boolean")
    if len(qkv.shape) != 3:
        raise ValueError("Convolution requires three-dimensional tiled input")
    batch = qkv.shape[1 if compact_input else 0]
    width = qkv.shape[-1]
    expected = (1, batch, width) if compact_input else (batch, 1, width)
    padded = (1, 32, width) if compact_input else (batch, 32, width)
    if not 1 <= batch <= 32 or width < sum(WIDTHS) or width % 32 or tuple(qkv.shape) != expected:
        raise ValueError("Require one token per user, batch 1..32, and tile-aligned packed width")
    if tuple(qkv.padded_shape) != padded or len(taps) != 4 or len(outputs) != 3:
        raise ValueError("Require exact padded geometry, four taps and three outputs")
    mesh = qkv.device()
    if "BLACKHOLE" not in str(mesh.arch()).upper():
        raise ValueError("Compact convolution currently targets Blackhole only")
    tensors = [qkv, history, *taps, *outputs]
    contracts = [
        (expected, ttnn.TILE_LAYOUT),
        ((batch, 3, sum(WIDTHS)), ttnn.ROW_MAJOR_LAYOUT),
        *[((1, 1, sum(WIDTHS)), ttnn.TILE_LAYOUT)] * 4,
        *[((1, batch, w), ttnn.TILE_LAYOUT) for w in WIDTHS],
    ]
    for tensor, (shape, layout) in zip(tensors, contracts):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != ttnn.bfloat16
            or tensor.layout != layout
            or tensor.memory_config() not in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)
            or tensor.device() != mesh
        ):
            raise ValueError("Convolution requires declared BF16 interleaved operand geometry")
    if tuple(history.padded_shape) != (batch, 3, sum(WIDTHS)) or any(
        tuple(t.padded_shape) != (1, 32, w) for t, w in zip(outputs, WIDTHS)
    ):
        raise ValueError("History and compact outputs require exact padded geometry")
    if len({(str(t.memory_config().buffer_type), t.buffer_address()) for t in tensors}) != len(tensors):
        raise ValueError("Convolution operands must not alias; history is the only in-place operand")
    grid = mesh.compute_with_storage_grid_size()
    assignments = work_items(sum(WIDTHS) // 32, grid.x, grid.y)
    cores = ttnn.num_cores_to_corerangeset(len(assignments), grid, row_wise=True)
    reader, writer, compute = (ttnn.RuntimeArgs() for _ in range(3))
    for x, y, first, stride, count in assignments:
        reader[x][y] = [*(t.buffer_address() for t in tensors[:6]), first, stride, count]
        writer[x][y] = [*(t.buffer_address() for t in outputs), first, stride, count]
        compute[x][y] = [count]

    def accessors(items):
        return [arg for t in items for arg in ttnn.TensorAccessorArgs(t).get_compile_time_args()]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=(HERE / name).read_text(),
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=ctargs,
            runtime_args=args,
            config=config,
        )
        for name, args, ctargs, config in (
            (
                "reader.cpp",
                reader,
                [batch, width // 32, int(compact_input), *accessors(tensors[:6])],
                ttnn.ReaderConfigDescriptor(),
            ),
            ("writer.cpp", writer, accessors(outputs), ttnn.WriterConfigDescriptor()),
            (
                "compute.cpp",
                compute,
                [],
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False
                ),
            ),
        )
    ]
    # All four activation windows remain resident until history writes finish.
    # CB5 is reader-private aligned face-row staging, never published.
    cbs = [
        ttnn.CBDescriptor(
            total_size=pages * 2048,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for cb, pages in enumerate((4, 1, 4, 2, 2, 2))
    ]
    return ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
