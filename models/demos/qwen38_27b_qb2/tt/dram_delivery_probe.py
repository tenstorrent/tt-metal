# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bank-local reads followed by credit-controlled unicast to remote workers.

This is a transport diagnostic, not an attention implementation. Every bank
has one dedicated reader and one consumer. Logical page order is retained in
the copied output; no production KV page table or attention math is modeled.
"""

from pathlib import Path

from .dram_read_probe import BANKS, PAGE_BYTES, PINNED_BANK_CORES, WORDS, assignments, validate_ring


def consumer_cores(placement):
    if placement == "near":
        return tuple((x + 1, y) for x, y in PINNED_BANK_CORES)
    if placement == "center":
        return tuple((4, y) for y in range(BANKS))
    if placement == "opposite":
        return tuple((10 if x == 0 else 2, y) for x, y in PINNED_BANK_CORES)
    raise ValueError("Consumer placement must be near, center or opposite")


def variants():
    base = dict(mode="bank_bulk", placement="pinned", packet_pages=8, depth=4, consumer_delay=0)
    rows = [{**base, "consumer_placement": p} for p in ("near", "center", "opposite")]
    rows += [{**rows[0], "depth": d} for d in (1, 2)]
    rows.append({**rows[0], "packet_pages": 15})
    return rows


def read(
    source,
    copied,
    receipt,
    *,
    mode,
    placement,
    packet_pages,
    depth,
    consumer_placement,
    consumer_delay=0,
    copy_payload=False,
):
    import ttnn

    validate_ring(packet_pages, depth)
    if mode != "bank_bulk" or placement != "pinned":
        raise ValueError("Delivery probe requires pinned bank-local bulk readers")
    if type(consumer_delay) is not int or not 0 <= consumer_delay <= 4096:
        raise ValueError("Consumer delay must be an integer in 0..4096")
    consumers = consumer_cores(consumer_placement)
    mesh = source.device()
    grid = mesh.compute_with_storage_grid_size()
    if "BLACKHOLE" not in str(mesh.arch()).upper() or mesh.dram_grid_size().x != BANKS:
        raise ValueError("Probe requires the allocated eight-bank Blackhole geometry")
    shape = tuple(source.shape)
    if len(shape) != 2 or shape[-1] != WORDS:
        raise ValueError("Source must be a matrix of 1088-byte raw pages")
    work = assignments(shape[0], mode, placement, (grid.x, grid.y))
    for tensor, wanted in ((source, shape), (copied, shape), (receipt, (BANKS, 8))):
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

    def core_set(coords):
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in coords})

    readers = core_set(PINNED_BANK_CORES)
    receivers = core_set(consumers)
    union = core_set((*PINNED_BANK_CORES, *consumers))
    read_args, send_args, receive_args = (ttnn.RuntimeArgs() for _ in range(3))
    for index, ((x, y, first, stride, count), (cx, cy)) in enumerate(zip(work, consumers)):
        sender = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        receiver = mesh.worker_core_from_logical_core(ttnn.CoreCoord(cx, cy))
        read_args[x][y] = [source.buffer_address(), first, stride, count, index % 4]
        send_args[x][y] = [receiver.x, receiver.y, count]
        receive_args[cx][cy] = [
            copied.buffer_address(),
            receipt.buffer_address(),
            first,
            stride,
            count,
            index,
            sender.x,
            sender.y,
        ]
    accessors = lambda tensors: [
        v for tensor in tensors for v in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
    ]
    here = Path(__file__).parent
    specs = [
        ("dram_read_probe_reader.cpp", readers, [2, packet_pages, depth, *accessors([source])], read_args, 0),
        ("dram_delivery_probe_sender.cpp", readers, [packet_pages, depth], send_args, 1),
        (
            "dram_delivery_probe_consumer.cpp",
            receivers,
            [packet_pages, depth, int(copy_payload), consumer_delay, *accessors([copied, receipt])],
            receive_args,
            1,
        ),
    ]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=(here / filename).read_text(),
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=args,
            runtime_args=runtime,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0 if processor == 0 else ttnn.DataMovementProcessor.RISCV_1,
                noc=ttnn.NOC.NOC_0 if processor == 0 else ttnn.NOC.NOC_1,
            ),
        )
        for filename, cores, args, runtime, processor in specs
    ]
    # A single descriptor over the union makes every slot's L1 address equal on
    # sender/receiver. Receiver slots are semaphore-owned raw storage; only the
    # sender's reader/forwarder uses the CB producer/consumer counters.
    cbs = [
        ttnn.CBDescriptor(
            total_size=packet_pages * PAGE_BYTES,
            core_ranges=union,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.uint32, page_size=PAGE_BYTES)],
        )
        for i in range(depth)
    ]
    cbs.append(
        ttnn.CBDescriptor(
            total_size=32,
            core_ranges=union,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=ttnn.uint32, page_size=32)],
        )
    )
    # Cumulative ready and consumed counters: no counter resets inside a replay,
    # no increment can be lost to a racing local reset. Dispatch initializes both.
    semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=union, initial_value=0) for i in range(2)]
    return ttnn.generic_op(
        [source, copied, receipt], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)
    )
