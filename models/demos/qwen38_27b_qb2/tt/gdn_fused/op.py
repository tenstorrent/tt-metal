# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Experimental GDN recurrence/epilogue with bounded, four-producer L1 handoff.

Separate compute cores preserve the two qualified math approximation modes.
This is a component experiment, not a model policy or serving default.
"""

import math
import struct
from pathlib import Path

from models.demos.qwen38_27b_qb2.tt.gdn_epilogue import op as epilogue
from models.demos.qwen38_27b_qb2.tt.gdn_step import op as recurrence

HERE = Path(__file__).parent


def schedule(heads, grid_x, grid_y, *, groups=None):
    """Each disjoint five-core group owns an increasing sequence of full heads."""
    if any(type(n) is not int or n < 1 for n in (heads, grid_x, grid_y)):
        raise ValueError("Positive integer head/grid dimensions required")
    maximum = min(heads, grid_x * grid_y // 5)
    groups = maximum if groups is None else groups
    if type(groups) is not int or not 1 <= groups <= maximum:
        raise ValueError("Require at least five cores and a fitting group count")

    def xy(index):
        return index % grid_x, index // grid_x

    return [
        dict(
            producers=[xy(g * 5 + p) for p in range(4)],
            consumer=xy(g * 5 + 4),
            first=g,
            stride=groups,
            count=(heads - 1 - g) // groups + 1,
        )
        for g in range(groups)
    ]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Qualified epilogue source anchor changed")
    return text.replace(old, new)


def consumer_source():
    """Retain qualified gate/weight loading and tile formatting verbatim."""
    text = epilogue.source("reader.cpp")
    text = replace_once(text, "    const auto raw = TensorAccessor(ra, get_arg_val<uint32_t>(0), 512);\n", "")
    text = replace_once(
        text,
        "    const uint32_t scratch = get_write_ptr(11);",
        """    const uint32_t scratch = get_write_ptr(11);
    const uint32_t mailbox = get_write_ptr(13);
    uint64_t credits[4];
    for (uint32_t p = 0; p < 4; ++p) {
        credits[p] = get_noc_addr(get_arg_val<uint32_t>(6 + p * 2),
                                  get_arg_val<uint32_t>(7 + p * 2), get_semaphore(4));
    }""",
    )
    text = replace_once(text, "        noc_async_read(raw.get_noc_addr(head), scratch, 512);\n", "")
    text = replace_once(
        text,
        "        const auto* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);",
        """        // Independent cumulative readiness prevents a fast partition from
        // satisfying the count for a missing partition of the current head.
        for (uint32_t p = 0; p < 4; ++p) {
            auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(p));
            noc_semaphore_wait_min(ready, item + 1);
        }
        const auto* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            mailbox + (item % 2) * 512);""",
    )
    text = replace_once(
        text,
        "        cb_push_back(dfb::gate, 4);\n    }\n}",
        """        cb_push_back(dfb::gate, 4);
        // Raw mailbox is now copied into consumer-owned CB storage. Compute
        // can continue after these credits release the mailbox for reuse.
        for (uint32_t p = 0; p < 4; ++p) {
            noc_semaphore_inc(credits[p], 1);
        }
    }
    noc_async_atomic_barrier();
}""",
    )
    return text


def fused_step(q, k, v, gates, state, raw, gate, weight, output, *, groups=None, capture_raw=False, epsilon=1e-6):
    """Opt-in B16/B32 TP-local component. State stays canonical FP32 DRAM.

    Raw is a caller-owned diagnostic buffer, written only with capture_raw.
    Each replay must initialize ready/consumed semaphores through dispatch.
    All addresses are rebuilt so independently allocated instances can rebind.
    """
    import ttnn

    if type(capture_raw) is not bool or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Require Boolean capture_raw and positive finite epsilon")
    mesh = state.device()
    total_heads = state.shape[0]
    if total_heads not in (16 * 12, 32 * 12) or "BLACKHOLE" not in str(mesh.arch()).upper():
        raise ValueError("Fused experiment requires Blackhole, B16/B32 and 12 local value heads")
    batch = total_heads // 12
    tensors = [q, k, v, gates, state, raw, gate, weight, output]
    shapes = [(batch * 4, 128)] * 2 + [
        (total_heads, 128),
        (total_heads, 8),
        (total_heads, 128, 128),
        (total_heads, 128),
        (1, batch, 4160),
        (128,),
        (1, batch, 1536),
    ]
    for i, (tensor, shape) in enumerate(zip(tensors, shapes)):
        layout = ttnn.TILE_LAYOUT if i == 4 or i >= 6 else ttnn.ROW_MAJOR_LAYOUT
        dtype = ttnn.float32 if i < 6 else ttnn.bfloat16
        memory = (ttnn.DRAM_MEMORY_CONFIG,) if i < 6 else (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.layout != layout
            or tensor.device() != mesh
            or tensor.memory_config() not in memory
        ):
            raise ValueError(f"Invalid fused operand {i}")
    if tuple(gate.padded_shape) != (1, 32, 4160) or tuple(output.padded_shape) != (1, 32, 1536):
        raise ValueError("Require exact compact gate/output padding")
    if len({(str(t.memory_config().buffer_type), t.buffer_address()) for t in tensors}) != len(tensors):
        raise ValueError("Operands must not alias; only state is updated in place")
    grid = mesh.compute_with_storage_grid_size()
    assignments = schedule(total_heads, grid.x, grid.y, groups=groups)

    def core_set(coords):
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in coords})

    producers = core_set([c for a in assignments for c in a["producers"]])
    consumers = core_set([a["consumer"] for a in assignments])
    union = core_set([c for a in assignments for c in (*a["producers"], a["consumer"])])
    pr, pw, pc, er, ew, ec = (ttnn.RuntimeArgs() for _ in range(6))
    for a in assignments:
        x, y = a["consumer"]
        receiver = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        producer_coords = []
        for part, (px, py) in enumerate(a["producers"]):
            physical = mesh.worker_core_from_logical_core(ttnn.CoreCoord(px, py))
            producer_coords.extend([physical.x, physical.y])
            first, stride, count = a["first"] * 4 + part, a["stride"] * 4, a["count"]
            pr[px][py] = [*(t.buffer_address() for t in tensors[:5]), first, stride, count]
            pw[px][py] = [
                state.buffer_address(),
                raw.buffer_address(),
                first,
                stride,
                count,
                receiver.x,
                receiver.y,
                part,
            ]
            pc[px][py] = [count]
        er[x][y] = [
            raw.buffer_address(),
            gate.buffer_address(),
            weight.buffer_address(),
            a["first"],
            a["stride"],
            a["count"],
            *producer_coords,
        ]
        ew[x][y] = [output.buffer_address(), a["first"], a["stride"], a["count"]]
        ec[x][y] = [a["count"]]

    def accessors(values):
        return [v for tensor in values for v in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()]

    rc = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=True)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for cb in range(9):
        modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    rc.unpack_to_dest_mode = modes
    eps = struct.unpack("I", struct.pack("f", epsilon))[0]
    specs = [
        (
            recurrence.kernel_source("reader.cpp"),
            producers,
            pr,
            [1, 3, *accessors(tensors[:5])],
            ttnn.ReaderConfigDescriptor(),
        ),
        (
            (HERE / "producer_writer.cpp").read_text(),
            producers,
            pw,
            [int(capture_raw), *accessors([state, raw])],
            ttnn.WriterConfigDescriptor(),
        ),
        (recurrence.kernel_source("compute_resident.cpp"), producers, pc, [1, 0], rc),
        (
            consumer_source(),
            consumers,
            er,
            [eps, 1, 12, 80, 1, *accessors([raw, gate, weight])],
            ttnn.ReaderConfigDescriptor(),
        ),
        (
            epilogue.source("writer.cpp"),
            consumers,
            ew,
            [1, 12, batch, *accessors([output])],
            ttnn.WriterConfigDescriptor(),
        ),
        (
            epilogue.source("compute.cpp"),
            consumers,
            ec,
            [1],
            ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True, fp32_dest_acc_en=True
            ),
        ),
    ]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            runtime_args=args,
            compile_time_args=ctargs,
            config=cfg,
        )
        for source, cores, args, ctargs, cfg in specs
    ]

    def cb(index, pages, dtype, cores):
        size = 4096 if dtype == ttnn.float32 else 2048
        return ttnn.CBDescriptor(
            total_size=pages * size,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
        )

    # One descriptor over the union guarantees the same mailbox address on
    # both roles. CB13 is semaphore-owned scratch; never use its CB tokens.
    cbs = [cb(13, 1, ttnn.float32, union)]
    cbs += [cb(i, n, ttnn.float32, producers) for i, n in enumerate(recurrence.circular_buffer_pages(4, 2))]
    formats = [
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.float32,
        ttnn.float32,
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.bfloat16,
    ]
    pages = [8, 8, 4, 4, 1, 1, 4, 8, 1, 1, 4, 1, 1]
    cbs += [cb(i, n, dtype, consumers) for i, (n, dtype) in enumerate(zip(pages, formats))]
    semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=union, initial_value=0) for i in range(5)]
    return ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores))
