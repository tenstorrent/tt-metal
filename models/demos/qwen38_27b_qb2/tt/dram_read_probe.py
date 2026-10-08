# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read-path calibration with BFP8-sized raw pages; no attention math or policy."""

from pathlib import Path

PAGE_BYTES = 1088
WORDS = PAGE_BYTES // 4
BANKS = 8
PINNED_BANK_CORES = ((0, 9), (0, 0), (0, 7), (0, 3), (7, 9), (7, 1), (7, 6), (7, 4))
MODES = {"interleaved_grid": 0, "bank_tiles": 1, "bank_bulk": 2}


def assignments(pages, mode, placement, grid=(12, 10)):
    """(x, y, first page, page stride, page count), disjoint complete coverage."""
    if type(pages) is not int or pages < BANKS or pages % BANKS:
        raise ValueError("Page count must be a positive multiple of eight banks")
    if mode not in MODES or placement not in ("grid", "row", "pinned"):
        raise ValueError("Unknown read mode or worker placement")
    gx, gy = grid
    if (gx, gy) != (12, 10):
        raise ValueError("This calibration pins the allocated Blackhole 12x10 grid")
    if mode == "interleaved_grid":
        if placement != "grid":
            raise ValueError("Interleaved-grid control uses the full worker grid")
        workers = min(pages, gx * gy)
        quotient, remainder = divmod(pages, workers)
        return [
            (i % gx, i // gx, i * quotient + min(i, remainder), 1, quotient + (i < remainder)) for i in range(workers)
        ]
    if placement == "grid":
        raise ValueError("Bank readers need one worker per bank")
    cores = PINNED_BANK_CORES if placement == "pinned" else tuple((i, 0) for i in range(BANKS))
    return [(x, y, bank, BANKS, pages // BANKS) for bank, (x, y) in enumerate(cores)]


def validate_ring(packet_pages, depth):
    if type(packet_pages) is not int or not 1 <= packet_pages <= 15:
        raise ValueError("Packets must contain 1..15 complete 1088-byte pages (<=16384 B)")
    if type(depth) is not int or depth not in (1, 2, 4, 8):
        raise ValueError("Ring depth must be 1, 2, 4 or 8; each slot has a distinct TRID")


def pattern_word(page, word, salt):
    return (page * 0x1F123BB5 + word * 0x45D9F3B + salt) & 0x7FFFFFFF


def expected_markers(assignment, packet_pages, salt):
    """Timed consumer checks both ends of every completed packet, in order."""
    _, _, first, stride, count = assignment
    a = b = weighted = blocks = 0
    for offset in range(0, count, packet_pages):
        n = min(packet_pages, count - offset)
        left = pattern_word(first + offset * stride, 0, salt)
        right = pattern_word(first + (offset + n - 1) * stride, WORDS - 1, salt)
        a = (a + left) & 0xFFFFFFFF
        b = (b + right) & 0xFFFFFFFF
        weighted = (weighted + (blocks + 1) * (left ^ right)) & 0xFFFFFFFF
        blocks += 1
    return [count, blocks, a, b, weighted, first, stride, 0xB17ECAFE]


def variants():
    rows = [dict(mode="interleaved_grid", placement="grid", packet_pages=15, depth=4)]
    rows += [dict(mode="bank_tiles", placement=p, packet_pages=15, depth=4) for p in ("row", "pinned")]
    rows.append(dict(mode="bank_bulk", placement="row", packet_pages=15, depth=4))
    rows += [dict(mode="bank_bulk", placement="pinned", packet_pages=15, depth=d) for d in (1, 2, 4, 8)]
    rows += [dict(mode="bank_bulk", placement="pinned", packet_pages=p, depth=4) for p in (1, 4, 8)]
    return rows


def read(source, copied, receipt, *, mode, placement, packet_pages, depth, copy_payload=False):
    """Read every source page once; copy mode checks bytes, timed mode checks markers.

    Inputs are UINT32 row-major raw pages of the same size as BFP8 tiles. No
    quantization, unpacking, attention math or model integration is implied.
    Caller owns all buffers and retains them across captured trace replay.
    """
    import ttnn

    validate_ring(packet_pages, depth)
    mesh = source.device()
    grid = mesh.compute_with_storage_grid_size()
    if "BLACKHOLE" not in str(mesh.arch()).upper() or mesh.dram_grid_size().x != BANKS:
        raise ValueError("Probe requires the allocated eight-bank Blackhole geometry")
    shape = tuple(source.shape)
    if len(shape) != 2 or shape[-1] != WORDS:
        raise ValueError("Source must be a matrix of 1088-byte raw pages")
    work = assignments(shape[0], mode, placement, (grid.x, grid.y))
    for tensor, wanted in ((source, shape), (copied, shape), (receipt, (len(work), 8))):
        if (
            tuple(tensor.shape) != wanted
            or tensor.dtype != ttnn.uint32
            or tensor.layout != ttnn.ROW_MAJOR_LAYOUT
            or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            or tensor.device() != mesh
        ):
            raise ValueError("Probe tensors require matching UINT32 row-major interleaved DRAM storage")
    if len({t.buffer_address() for t in (source, copied, receipt)}) != 3:
        raise ValueError("Source, copy output and receipt must not alias")
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y, *_ in work})
    read_args, write_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for index, (x, y, first, stride, count) in enumerate(work):
        read_args[x][y] = [source.buffer_address(), first, stride, count, index % 4]
        write_args[x][y] = [copied.buffer_address(), receipt.buffer_address(), first, stride, count, index]
    accessors = lambda tensors: [v for t in tensors for v in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
    here = Path(__file__).parent
    read_config = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)
    write_config = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=(here / filename).read_text(),
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=compile_args,
            runtime_args=runtime_args,
            config=config,
        )
        for filename, compile_args, runtime_args, config in (
            (
                "dram_read_probe_reader.cpp",
                [MODES[mode], packet_pages, depth, *accessors([source])],
                read_args,
                read_config,
            ),
            (
                "dram_read_probe_writer.cpp",
                [packet_pages, depth, int(copy_payload), *accessors([copied, receipt])],
                write_args,
                write_config,
            ),
        )
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=packet_pages * PAGE_BYTES,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.uint32, page_size=PAGE_BYTES)],
        )
        for i in range(depth)
    ]
    cbs.append(
        ttnn.CBDescriptor(
            total_size=32,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=ttnn.uint32, page_size=32)],
        )
    )
    return ttnn.generic_op([source, copied, receipt], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
