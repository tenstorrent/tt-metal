# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DRAM read + NoC forward bandwidth on one Blackhole chip: the 510 GB/s raw-read recipe (bank-contiguous source, one
BRISC/NOC0 reader on each bank's NOC0-optimal core + one NCRISC/NOC1 reader on its NOC1-optimal core, 16 readers on
15 cores) with every chunk forwarded to compute cores over the reader's own NoC (kernels/dram_read_fwd.cpp).

Forwarding modes (receivers = the grid cores that are not readers; scratch = an L1-sharded tensor on all of them):
  none         read only (baseline)
  uni{R}       each reader owns the R free cores closest ALONG ITS NoC (NOC0 flows east/south, NOC1 west/north; both
               wrap) and sends every chunk to one of them, round-robin
               (N-sharded weights: every byte lands on exactly one compute core)
  mcast_row    each chunk multicast to the reader's row of the compute block downstream on its NoC (5 or 4 cores)
  mcast_side   each chunk multicast to that whole compute block (5 x 10 or 4 x 10 cores)
Compute blocks: logical columns strictly between the two reader columns, and those right of the second one.

Tag ``dramfwd_{mode}_ch{chunk}_half{KB}``; read GB/s = bytes read / device kernel time, delivered GB/s = bytes landed
in receivers (x destinations for multicast) / time (analyze_dram_read_fwd.py).
"""

import json
import os
from pathlib import Path

import pytest
from loguru import logger

import ttnn

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None


def _env_list(name, default, cast=str):
    return [cast(v) for v in os.environ.get(name, default).split(",") if v]


KERNEL = "models/demos/mimo_v2_d_p/tests/perf/kernels/dram_read_fwd.cpp"
MODES = _env_list("MIMO_FWD_MODES", "none,uni1,uni2,uni4,uni5,mcast_row,mcast_side")
CHUNKS = _env_list("MIMO_FWD_CHUNKS", "4096,16384", int)
HALF_KB = _env_list("MIMO_FWD_HALF_KB", "64,128", int)
ITERS = int(os.environ.get("MIMO_FWD_ITERS", "3"))
STATS_PATH = Path(os.environ.get("MIMO_FWD_STATS", "generated/mimo_dram_fwd/cases.jsonl"))
H, W = 8192, 16384  # 256 MB of bf16, DRAM width-sharded (one contiguous shard per bank)
DST_SLOTS = 4
NOC_X, NOC_Y = 17, 12  # Blackhole NoC torus (translated worker coords live inside it)


def noc_hops(src, dst, noc):
    """Hops from src to dst along one NoC torus: NOC0 travels +x then +y, NOC1 -x then -y (both wrap)."""
    if noc == 0:
        return (dst.x - src.x) % NOC_X + (dst.y - src.y) % NOC_Y
    return (src.x - dst.x) % NOC_X + (src.y - dst.y) % NOC_Y


def downstream_block(reader_x, reader_cols, noc, grid_x):
    """Logical columns of the compute block that lies downstream of a reader column on this NoC: NOC0 (eastward)
    takes the block right after it, NOC1 (westward, wrapping) the block right before it."""
    left = [x for x in range(grid_x) if reader_cols[0] < x < reader_cols[1]]
    right = [x for x in range(grid_x) if x > reader_cols[1]]
    if noc == 0:
        return left if reader_x == reader_cols[0] else right
    return right if reader_x == reader_cols[0] else left


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("half_kb", HALF_KB, ids=lambda h: f"half{h}")
@pytest.mark.parametrize("chunk", CHUNKS, ids=lambda c: f"ch{c}")
@pytest.mark.parametrize("mode", MODES)
def test_dram_read_fwd(device, mode, chunk, half_kb):
    half = half_kb * 1024
    if half % chunk:
        pytest.skip("half must be a multiple of chunk")
    banks = device.dram_grid_size().x
    grid = device.compute_with_storage_grid_size()
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    spec = ttnn.ShardSpec(dram_grid, (H, W // banks), ttnn.ShardOrientation.ROW_MAJOR)
    src = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, H, W]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec),
    )
    total = H * W * 2

    # Readers: (logical core, risc, bank); BRISC on NOC0-optimal, NCRISC on NOC1-optimal cores.
    opt = {
        r: list(device.get_optimal_dram_bank_to_logical_worker_assignment(n))
        for r, n in (("brisc", ttnn.NOC.NOC_0), ("ncrisc", ttnn.NOC.NOC_1))
    }
    readers = [(opt[r][b], r, b) for b in range(banks) for r in ("brisc", "ncrisc")]
    reader_xy = {(c.x, c.y) for c, _, _ in readers}
    free = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in reader_xy]
    phys = lambda c: device.worker_core_from_logical_core(c)
    reader_cols = sorted({c.x for c, _, _ in readers})

    assert len(reader_cols) == 2, reader_cols

    # Receiver scratch: one L1 shard per free core, same address everywhere.
    free_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in free])
    shard_words = DST_SLOTS * chunk // 4
    scratch = ttnn.allocate_tensor_on_device(
        ttnn.Shape([len(free), shard_words]),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(free_set, (1, shard_words), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    dst_addr = scratch.buffer_address()

    fwd = 0 if mode == "none" else (1 if mode.startswith("uni") else 2)
    per_reader_bytes = total // len(readers) // half * half
    rt = {r: ttnn.RuntimeArgs() for r in ("brisc", "ncrisc")}
    taken, delivered = set(), 0
    for c, r, b in readers:
        k = sum(1 for c2, r2, b2 in readers if b2 == b and (r2, c2.x, c2.y) < (r, c.x, c.y))  # reader index on bank
        args = [src.buffer_address(), b, k * per_reader_bytes, per_reader_bytes, dst_addr]
        if fwd == 1:
            nrecv = int(mode[3:])
            noc = 0 if r == "brisc" else 1
            cand = sorted(
                (f for f in free if (f.x, f.y) not in taken), key=lambda f: (noc_hops(phys(c), phys(f), noc), f.y, f.x)
            )[:nrecv]
            assert len(cand) == nrecv, f"not enough free cores for {nrecv} receivers per reader"
            taken |= {(f.x, f.y) for f in cand}
            args += [nrecv] + [(phys(f).x << 16) | phys(f).y for f in cand]
            delivered += per_reader_bytes
        elif fwd == 2:
            cols = downstream_block(c.x, reader_cols, 0 if r == "brisc" else 1, grid.x)
            rows = [c.y] if mode == "mcast_row" else list(range(grid.y))
            lo = phys(ttnn.CoreCoord(min(cols), min(rows)))
            hi = phys(ttnn.CoreCoord(max(cols), max(rows)))
            n_dest = len(cols) * len(rows)
            # NOC0 multicast goes start=(low) -> end=(high); NOC1 travels the other way.
            rect = [lo.x, lo.y, hi.x, hi.y] if r == "brisc" else [hi.x, hi.y, lo.x, lo.y]
            args += rect + [n_dest]
            delivered += per_reader_bytes * n_dest
        rt[r][c.x][c.y] = args
    read_bytes = per_reader_bytes * len(readers)

    kernels, cbs = [], []
    for cb_idx, (r, proc, noc) in enumerate(
        (
            ("brisc", ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
            ("ncrisc", ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
        )
    ):
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, r2, _ in readers if r2 == r])
        cbs.append(
            ttnn.CBDescriptor(
                total_size=2 * half,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=cb_idx, data_format=ttnn.bfloat16, page_size=chunk)
                ],
            )
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores,
                compile_time_args=[chunk, half, cb_idx, fwd, DST_SLOTS],
                runtime_args=rt[r],
                config=ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc),
            )
        )
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    tag = f"dramfwd_{mode}_ch{chunk}_half{half_kb}"
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("a") as f:
        f.write(
            json.dumps(
                {
                    "tag": tag,
                    "mode": mode,
                    "chunk": chunk,
                    "half_kb": half_kb,
                    "read_bytes": read_bytes,
                    "delivered_bytes": delivered,
                    "receivers": len(taken) if fwd == 1 else None,
                }
            )
            + "\n"
        )
    for it in range(1 + ITERS):
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_start")
        ttnn.generic_op([src, scratch], program)
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_end")
    logger.info(f"ran {tag}: read {read_bytes / 1e6:.0f} MB, delivered {delivered / 1e6:.0f} MB")


SPLIT_KERNEL = "models/demos/mimo_v2_d_p/tests/perf/kernels/dram_split.cpp"
SPLIT_MODES = _env_list("MIMO_SPLIT_MODES", "none,uni1,uni2,uni5,uni10,mcast_row,mcast_side")
PAIR = os.environ.get("MIMO_SPLIT_PAIR", "0") == "1"  # 2 NOC0 readers per bank (16 cores) instead of 1


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("half_kb", HALF_KB, ids=lambda h: f"half{h}")
@pytest.mark.parametrize("chunk", CHUNKS, ids=lambda c: f"ch{c}")
@pytest.mark.parametrize("mode", SPLIT_MODES)
def test_dram_read_fwd_split(device, mode, chunk, half_kb):
    """Split roles on each bank's NOC0-optimal core: BRISC reads the bank over NOC0 into a CB (half_kb per batch,
    2 batches deep), NCRISC drains it and forwards over NOC1 (kernels/dram_split.cpp). Tag ``dramsplit_...``."""
    half = half_kb * 1024
    if half % chunk:
        pytest.skip("half must be a multiple of chunk")
    batch = half // chunk
    banks = device.dram_grid_size().x
    grid = device.compute_with_storage_grid_size()
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    spec = ttnn.ShardSpec(dram_grid, (H, W // banks), ttnn.ShardOrientation.ROW_MAJOR)
    src = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, H, W]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec),
    )
    readers = list(device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0))  # reader for bank b
    if PAIR:  # a second NOC0 reader per bank on the east neighbour of each optimal core (raw reads: 510 GB/s)
        readers += [ttnn.CoreCoord(c.x + 1, c.y) for c in readers]
    reader_cols = sorted({c.x for c in readers[:banks]})
    assert len(reader_cols) == 2, reader_cols

    reader_xy = {(c.x, c.y) for c in readers}
    free = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in reader_xy]
    free_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in free])
    shard_words = DST_SLOTS * chunk // 4
    scratch = ttnn.allocate_tensor_on_device(
        ttnn.Shape([len(free), shard_words]),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(free_set, (1, shard_words), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    phys = lambda c: device.worker_core_from_logical_core(c)
    fwd = 0 if mode == "none" else (1 if mode.startswith("uni") else 2)
    per_bank_readers = len(readers) // banks
    per_reader = H * W * 2 // banks // per_bank_readers // half * half
    rd_rt, fw_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    taken, delivered = set(), 0
    for i, c in enumerate(readers):
        rd_rt[c.x][c.y] = [src.buffer_address(), i % banks, (i // banks) * per_reader, per_reader]
        args = [per_reader, scratch.buffer_address()]
        if fwd == 1:
            n = int(mode[3:])
            cand = sorted(
                (f for f in free if (f.x, f.y) not in taken), key=lambda f: (noc_hops(phys(c), phys(f), 1), f.y, f.x)
            )[:n]
            assert len(cand) == n, f"not enough free cores for {n} receivers per reader"
            taken |= {(f.x, f.y) for f in cand}
            args += [n] + [(phys(f).x << 16) | phys(f).y for f in cand]
            delivered += per_reader
        elif fwd == 2:
            cols = downstream_block(c.x, reader_cols, 1, grid.x)
            rows = [c.y] if mode == "mcast_row" else list(range(grid.y))
            lo, hi = phys(ttnn.CoreCoord(min(cols), min(rows))), phys(ttnn.CoreCoord(max(cols), max(rows)))
            args += [hi.x, hi.y, lo.x, lo.y, len(cols) * len(rows)]  # the forwarder multicasts on NOC1
            delivered += per_reader * len(cols) * len(rows)
        fw_rt[c.x][c.y] = args
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in readers])
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=SPLIT_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[0, chunk, batch, 0, fwd, DST_SLOTS],
            runtime_args=rd_rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        )
    ]
    if PAIR and fwd == 2:
        pytest.skip("paired readers sit inside the multicast blocks; multicast rectangles would cover readers")
    if fwd:
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=SPLIT_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores,
                compile_time_args=[1, chunk, batch, 0, fwd, DST_SLOTS],
                runtime_args=fw_rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
                ),
            )
        )
    else:  # no consumer: let the reader free its own pages
        pass
    cb = ttnn.CBDescriptor(
        total_size=2 * half,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=chunk)],
    )
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb])
    if not fwd:
        pytest.skip(
            "reader-only baseline = test_dram_read_bw wsh c8 brisc optimal (478 GB/s); a CB with no consumer would block"
        )

    tag = f"dramsplit{'2' if PAIR else ''}_{mode}_ch{chunk}_half{half_kb}"
    read_bytes = per_reader * len(readers)
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("a") as f:
        f.write(
            json.dumps(
                {
                    "tag": tag,
                    "mode": ("split2_" if PAIR else "split_") + mode,
                    "chunk": chunk,
                    "half_kb": half_kb,
                    "read_bytes": read_bytes,
                    "delivered_bytes": delivered,
                    "receivers": len(taken) if fwd == 1 else None,
                }
            )
            + "\n"
        )
    for it in range(1 + ITERS):
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_start")
        ttnn.generic_op([src, scratch], program)
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_end")
    logger.info(f"ran {tag}: read {read_bytes / 1e6:.0f} MB, delivered {delivered / 1e6:.0f} MB")
