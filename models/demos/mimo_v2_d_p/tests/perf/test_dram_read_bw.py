# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Raw DRAM read bandwidth on one Blackhole chip: no compute, just 'download the weights' into L1 and discard.

A generic_op runs kernels/dram_read_bw.cpp on C Tensix cores, on BRISC (NOC0), NCRISC (NOC1) or both (each core's
share split in half). The source is a 256 MB DRAM tensor in one of these layouts:

  il_tile_bf16 / il_tile_bf8   interleaved TILE pages (2048 / 1088 B): every page is its own read, round-robin banks
  il_rm_4k / il_rm_32k         interleaved ROW_MAJOR with 4 KB / 32 KB pages (one row = one page)
  wsh                          DRAM WIDTH_SHARDED, one shard per bank; read bank-contiguously with CHUNK-byte reads
  nd_tile / nd_128k / nd_2m    ND-sharded (NdShardSpec, round-robin over banks) with 2 KB / 128 KB / 2 MB shards;
                               read bank-contiguously, one read per shard (capped at 64 KB per request)

For the bank-contiguous layouts the C cores are split evenly over the banks (C / banks readers per bank, each taking
a contiguous slice of that bank); ``assign=optimal`` / ``optimal_noc1`` use the device's optimal DRAM-bank -> worker map
for NOC0 / NOC1 (8 cores), ``optimal_split`` runs BRISC on each bank's NOC0-optimal core and NCRISC on its NOC1-optimal core.
Bytes in flight per RISC = MIMO_DRAM_RING_KB (default 256): a read barrier is taken every time that L1 ring wraps.

Tag ``dramrd_{layout}_c{C}_{risc}_ch{chunk}_{assign}``; GB/s = 256 MB / device kernel time
(analyze_dram_read_bw.py).
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


KERNEL = "models/demos/mimo_v2_d_p/tests/perf/kernels/dram_read_bw.cpp"
LAYOUTS = _env_list("MIMO_DRAM_LAYOUTS", "il_tile_bf16,il_tile_bf8,il_rm_4k,il_rm_32k,wsh,nd_tile,nd_128k,nd_2m")
CORES = _env_list("MIMO_DRAM_CORES", "8,16,32,64,all")
RISCS = _env_list("MIMO_DRAM_RISCS", "brisc,ncrisc,both")
WSH_CHUNKS = _env_list("MIMO_DRAM_CHUNKS", "2048,8192,16384,65536", int)
RING_BYTES = int(os.environ.get("MIMO_DRAM_RING_KB", "256")) * 1024
ITERS = int(os.environ.get("MIMO_DRAM_ITERS", "3"))
STATS_PATH = Path(os.environ.get("MIMO_DRAM_STATS", "generated/mimo_dram_read/cases.jsonl"))
H, W = 8192, 16384  # 256 MB of bf16


def _make_source(device, layout, banks):
    """(tensor, mode, chunk bytes or None for 'per-shard', per-bank bytes) for a 256 MB source."""
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    shape, dtype, lay, mem, chunk = (
        ttnn.Shape([1, 1, H, W]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
        None,
    )
    if layout == "il_tile_bf16":
        chunk = 2048
    elif layout == "il_tile_bf8":
        dtype, chunk = ttnn.bfloat8_b, 1088
    elif layout == "il_rm_4k":
        shape, lay, chunk = ttnn.Shape([1, 1, H * W // 2048, 2048]), ttnn.ROW_MAJOR_LAYOUT, 4096
    elif layout == "il_rm_32k":
        lay, chunk = ttnn.ROW_MAJOR_LAYOUT, 2 * W
    elif layout == "wsh":
        spec = ttnn.ShardSpec(dram_grid, (H, W // banks), ttnn.ShardOrientation.ROW_MAJOR)
        mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)
    else:
        shard = {"nd_tile": [32, 32], "nd_128k": [32, 2048], "nd_2m": [1024, 1024]}[layout]
        mem = ttnn.MemoryConfig(ttnn.BufferType.DRAM, ttnn.NdShardSpec(ttnn.Shape([1, 1] + shard), dram_grid))
        chunk = min(shard[0] * shard[1] * 2, 65536)
    t = ttnn.allocate_tensor_on_device(shape, dtype, lay, device, mem)
    mode = 0 if layout.startswith("il_") else 1
    total = H * W * 2 if dtype == ttnn.bfloat16 else H * W * 1088 // 1024
    return t, mode, chunk, total


def _cores(device, n, assign, banks):
    grid = device.compute_with_storage_grid_size()
    if assign == "noc0_pair":  # each bank's NOC0-optimal core followed by its east neighbour (both read on NOC0)
        opt = list(device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0))
        return opt + [ttnn.CoreCoord(c.x + 1, c.y) for c in opt]
    if assign in ("optimal", "optimal_noc1"):
        noc = ttnn.NOC.NOC_1 if assign == "optimal_noc1" else ttnn.NOC.NOC_0
        return list(device.get_optimal_dram_bank_to_logical_worker_assignment(noc))
    allc = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    n = len(allc) if n == "all" else int(n)
    return allc[:n]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("assign", _env_list("MIMO_DRAM_ASSIGN", "grid"))
@pytest.mark.parametrize("risc", RISCS)
@pytest.mark.parametrize("cores", CORES, ids=lambda c: f"c{c}")
@pytest.mark.parametrize("chunk", [0] + WSH_CHUNKS, ids=lambda c: f"ch{c}")
@pytest.mark.parametrize("layout", LAYOUTS)
def test_dram_read_bw(device, layout, chunk, cores, risc, assign):
    banks = device.dram_grid_size().x
    if (layout == "wsh") == (chunk == 0):
        pytest.skip("chunk sweep only applies to the width-sharded layout")
    if assign == "noc0_pair" and (cores != "16" or risc != "brisc"):
        pytest.skip("noc0_pair = 2 BRISC/NOC0 readers per bank (16 cores)")
    if assign not in ("grid", "noc0_pair") and cores != "8":
        pytest.skip("optimal assignments are one core per bank")
    if assign == "optimal_split" and risc != "both":
        pytest.skip("optimal_split = BRISC on each bank's NOC0-optimal core + NCRISC on its NOC1-optimal core")
    src, mode, natural_chunk, total = _make_source(device, layout, banks)
    chunk = chunk or natural_chunk
    riscs = ["brisc", "ncrisc"] if risc == "both" else [risc]
    if assign == "optimal_split":
        per_risc = {
            "brisc": _cores(device, cores, "optimal", banks),
            "ncrisc": _cores(device, cores, "optimal_noc1", banks),
        }
    else:
        core_list = _cores(device, cores, assign, banks)
        if mode == 1:
            core_list = core_list[: len(core_list) // banks * banks]
        per_risc = {r: core_list for r in riscs}
    core_list = sorted({(c.x, c.y) for r in riscs for c in per_risc[r]})
    core_list = [ttnn.CoreCoord(x, y) for x, y in core_list]
    ncores = len(core_list)
    rangeset = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in core_list])
    risc_ranges = {r: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in per_risc[r]]) for r in riscs}
    ring = RING_BYTES // chunk * chunk
    cfg = {
        "brisc": (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
        "ncrisc": (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
    }

    # Work split: interleaved -> contiguous page ranges over all readers; bank-contiguous -> readers per bank.
    # Reader i of a bank-contiguous layout serves bank i % banks, so interleave the RISCs per core list position.
    # optimal_split: both RISCs of position i serve bank i; otherwise reader index -> bank.
    readers = [(per_risc[r][i], r, i) for i in range(len(per_risc[riscs[0]])) for r in riscs]
    bank_of = (lambda idx, pos: pos % banks) if assign == "optimal_split" else (lambda idx, pos: idx % banks)
    rt = {r: ttnn.RuntimeArgs() for r in riscs}
    base = src.buffer_address()
    if mode == 0:
        num_pages = total // chunk
        per, extra = divmod(num_pages, len(readers))
        start = 0
        for i, (c, r, _) in enumerate(readers):
            n = per + (i < extra)
            rt[r][c.x][c.y] = [base, start, n]
            start += n
    else:
        bank_bytes = total // banks
        per_bank = len(readers) // banks
        slice_bytes = bank_bytes // per_bank // chunk * chunk
        seen = [0] * banks
        for i, (c, r, pos) in enumerate(readers):
            bank = bank_of(i, pos)
            k, seen[bank] = seen[bank], seen[bank] + 1
            rt[r][c.x][c.y] = [base, bank, k * slice_bytes, slice_bytes]
        total = slice_bytes * len(readers)  # bytes actually read

    kernels, cbs = [], []
    for cb_idx, r in enumerate(riscs):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=ring,
                core_ranges=rangeset,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=cb_idx, data_format=ttnn.bfloat16, page_size=chunk)
                ],
            )
        )
        proc, noc = cfg[r]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=risc_ranges[r],
                compile_time_args=[mode, chunk, ring, banks, cb_idx],
                runtime_args=rt[r],
                config=ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc),
            )
        )
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    tag = f"dramrd_{layout}_c{ncores}_{risc}_ch{chunk}_{assign}"
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("a") as f:
        f.write(
            json.dumps(
                {
                    "tag": tag,
                    "layout": layout,
                    "cores": ncores,
                    "risc": risc,
                    "chunk": chunk,
                    "assign": assign,
                    "bytes": total,
                    "ring": ring,
                }
            )
            + "\n"
        )
    for it in range(1 + ITERS):
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_start")
        ttnn.generic_op([src, src], program)
        ttnn.synchronize_device(device)
        if it:
            signpost(f"{tag}_end")
    logger.info(f"ran {tag}: {total / 1e6:.0f} MB")
