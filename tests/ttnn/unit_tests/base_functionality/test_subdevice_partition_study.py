# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Spatial partitioning of one Blackhole chip into two sub-devices: what it buys, measured.

(a) overlap of a compute-bound and of a DRAM-bound kernel pair on two disjoint sub-devices,
(b) global-semaphore handoff from sub-device 0 to sub-device 1,
(c) global-circular-buffer streaming from sub-device 0 senders to sub-device 1 receivers,
(d) two co-hosted ttnn matmul workloads: partitioned and concurrent vs time-sliced on the full grid,
(e) what an op left on its default (full) grid does while the chip is partitioned.

Every test checks correctness. Timings are logged and, when SUBDEVICE_STUDY_RESULTS names a directory,
written there as <test id>.json; they never fail a test, except the compute-bound overlap ratio in (a).
"""

import json
import os
import re
import statistics
import time
from contextlib import contextmanager

import numpy as np
import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_py_test_utils_common.utility_functions import run_for_blackhole, skip_for_slow_dispatch

pytestmark = [run_for_blackhole("calibrated for Blackhole"), skip_for_slow_dispatch("sub-devices need fast dispatch")]

KERNELS = "tests/ttnn/unit_tests/base_functionality/kernels/subdevice_study"
STATS_WORDS = 16  # one 64-byte stats page per core
CB_STATS = 8
TILE_BYTES = 2048  # bf16 32x32
MIN_COMPUTE_OVERLAP = 1.5


# ----------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------


def _record(request, data):
    """Log `data` and, if SUBDEVICE_STUDY_RESULTS is set, write it as JSON named after the test id."""
    logger.info(f"{request.node.name}: {json.dumps(data, indent=1)}")
    out_dir = os.environ.get("SUBDEVICE_STUDY_RESULTS")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(data, f, indent=1)


def _crs(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


def _cores(crs):
    """Cores of a CoreRangeSet in row-major order."""
    out = []
    for r in crs.ranges():
        for y in range(r.start.y, r.end.y + 1):
            for x in range(r.start.x, r.end.x + 1):
                out.append(ttnn.CoreCoord(x, y))
    return out


def _halves(device, split=None):
    """Disjoint left/right column blocks of the worker grid, split at column `split` (default: the middle,
    11x10 -> cols 0-4 | 5-10; 13x10 -> 0-5 | 6-12)."""
    g = device.compute_with_storage_grid_size()
    split = split or g.x // 2
    return _crs(0, 0, split - 1, g.y - 1), _crs(split, 0, g.x - 1, g.y - 1)


def _desc(crs):
    r = list(crs.ranges())[0]
    return f"({r.start.x},{r.start.y})-({r.end.x},{r.end.y}) {crs.num_cores()} cores"


@contextmanager
def _sub_devices(device, core_sets):
    mgr = device.create_sub_device_manager([ttnn.SubDevice([c]) for c in core_sets], 0)
    device.load_sub_device_manager(mgr)
    try:
        yield [ttnn.SubDeviceId(i) for i in range(len(core_sets))]
    finally:
        device.reset_sub_device_stall_group()
        ttnn.synchronize_device(device)
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(mgr)


def _launch_pair(device, sd, consumer_on_sd1, producer_on_sd0):
    """Launch a consumer that spins on sub-device 1, then its producer on sub-device 0, and wait for both.

    A program's first launch uploads its kernel binary, which is a buffer write and so stalls on every
    sub-device in the stall group; if that included sub-device 1, the producer would wait for the consumer
    that waits for it (a deadlock observed on this build). Hence the stall group around the producer launch.
    """
    consumer_on_sd1()
    device.set_sub_device_stall_group([sd[0]])
    producer_on_sd0()
    device.reset_sub_device_stall_group()
    ttnn.synchronize_device(device)


def _kernel(name, cores, config, rt, named=(), ct=()):
    return ttnn.KernelDescriptor(
        kernel_source=f"{KERNELS}/{name}",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=list(ct),
        named_compile_time_args=list(named),
        runtime_args=rt,
        config=config,
    )


def _cb(index, cores, page_size, num_pages, dtype):
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)
    return ttnn.CBDescriptor(total_size=page_size * num_pages, core_ranges=cores, format_descriptors=[fmt])


def _remote_cb(index, cores, page_size, gcb, dtype=ttnn.uint32):
    cb = ttnn.CBDescriptor()
    cb.total_size = gcb.size()
    cb.core_ranges = cores
    cb.remote_format_descriptors = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)]
    cb.set_global_circular_buffer(gcb)
    return cb


def _taa(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _u32_tensor(device, data):
    """Row-major uint32 DRAM tensor from an int64/uint32 numpy array (one row = one page)."""
    t = torch.from_numpy(np.ascontiguousarray(data.astype(np.uint32).view(np.int32)))
    return ttnn.from_torch(
        t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _u32_zeros(device, rows, words):
    return _u32_tensor(device, np.zeros((rows, words), dtype=np.uint32))


def _u32_read(t):
    return ttnn.to_torch(t).view(torch.int32).numpy().astype(np.int64) & 0xFFFFFFFF


def _ticks(stats, col, ref):
    """Wall-clock ticks in column `col` relative to tick `ref`, from the low 32-bit word only (signed, +-1.5 s).

    The kernels store the high word as well, but get_timestamp()'s two 32-bit reads are not atomic across a
    low-word wrap on this build: a spin kernel saw its 64-bit clock jump by exactly 2^32.
    """
    return ((stats[:, col] - ref + 2**31) & 0xFFFFFFFF) - 2**31


def _timed(fn, sync, reps):
    """Median host wall time (s) of fn() + sync() over reps runs."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times


def _spin_program(core, stats, ticks):
    """One-core program that spins `ticks` wall-clock ticks and writes [t_start, t_end] to stats page 0."""
    crs = _crs(core.x, core.y, core.x, core.y)
    k = _kernel(
        "spin_ticks.cpp",
        crs,
        ttnn.WriterConfigDescriptor(),
        [(core, [ticks, stats.buffer_address()])],
        [("cb_stats", CB_STATS)],
        _taa(stats),
    )
    return ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(CB_STATS, crs, 64, 1, ttnn.uint32)])


_CLOCK_HZ = {}


def _wall_clock_hz(device):
    """Device wall-clock frequency: spins of two lengths, alternated and timed on the host, so the fixed
    launch + sync overhead cancels. Median over rounds; a single host-timed pair was seen to be off by 10x."""
    if "hz" in _CLOCK_HZ:
        return _CLOCK_HZ["hz"]
    stats = _u32_zeros(device, 1, STATS_WORDS)
    sync = lambda: ttnn.synchronize_device(device)
    progs = {t: _spin_program(ttnn.CoreCoord(0, 0), stats, t) for t in (100_000_000, 600_000_000)}

    def sample(ticks):
        t0 = time.perf_counter()
        ttnn.generic_op([stats, stats], progs[ticks])
        sync()
        host = time.perf_counter() - t0
        st = _u32_read(stats)
        return host, int(_ticks(st, 2, int(st[0, 0]))[0])

    for t in progs:  # compile and warm the chip up
        sample(t)
    estimates = []
    for _ in range(5):
        (ha, na), (hb, nb) = sample(100_000_000), sample(600_000_000)
        estimates.append((nb - na) / (hb - ha))
    hz = statistics.median(estimates)
    logger.info(f"device wall clock: {hz / 1e6:.1f} MHz (estimates {[round(e / 1e6, 1) for e in estimates]})")
    assert 0.5e9 < hz < 2.0e9 and max(estimates) - min(estimates) < 0.05 * hz, f"unstable clock estimates {estimates}"
    _CLOCK_HZ["hz"] = hz
    return hz


def _interval_overlap(a, b):
    """Fraction of the shorter interval covered by the other ([start, end] pairs)."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1, min(a[1] - a[0], b[1] - b[0]))


# ----------------------------------------------------------------------------------------------------------
# (a) Overlap proof
# ----------------------------------------------------------------------------------------------------------


def _compute_program(cores, ab, out, stats, iters):
    coords = _cores(cores)
    cbs = [
        _cb(0, cores, TILE_BYTES, 1, ttnn.bfloat16),
        _cb(1, cores, TILE_BYTES, 1, ttnn.bfloat16),
        _cb(16, cores, TILE_BYTES, 1, ttnn.bfloat16),
        _cb(CB_STATS, cores, 64, 1, ttnn.uint32),
    ]
    reader = _kernel(
        "reader_ab_tiles.cpp",
        cores,
        ttnn.ReaderConfigDescriptor(),
        [(c, [ab.buffer_address()]) for c in coords],
        [("cb_a", 0), ("cb_b", 1)],
        _taa(ab),
    )
    compute = _kernel(
        "compute_matmul_loop.cpp",
        cores,
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4),
        [(c, [iters]) for c in coords],
        [("cb_a", 0), ("cb_b", 1), ("cb_out", 16), ("inner", 4)],
    )
    writer = _kernel(
        "writer_tile_and_stats.cpp",
        cores,
        ttnn.WriterConfigDescriptor(),
        [(c, [out.buffer_address(), stats.buffer_address(), i]) for i, c in enumerate(coords)],
        [("cb_out", 16), ("cb_stats", CB_STATS)],
        _taa(out) + _taa(stats),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, compute, writer], semaphores=[], cbs=cbs)


def _dram_program(cores, src, stats, pages_per_core, passes, full_check, page_size, batch=8, depth=4):
    coords = _cores(cores)
    cbs = [_cb(0, cores, page_size, batch * depth, ttnn.uint32), _cb(CB_STATS, cores, 64, 1, ttnn.uint32)]
    named = [("cb_data", 0), ("batch", batch), ("page_size", page_size)]
    reader = _kernel(
        "reader_dram_stream.cpp",
        cores,
        ttnn.ReaderConfigDescriptor(),
        [(c, [src.buffer_address(), i * pages_per_core, pages_per_core, passes]) for i, c in enumerate(coords)],
        named,
        _taa(src),
    )
    consumer = _kernel(
        "consumer_checksum.cpp",
        cores,
        ttnn.WriterConfigDescriptor(),
        [(c, [stats.buffer_address(), i, pages_per_core, passes]) for i, c in enumerate(coords)],
        named + [("cb_stats", CB_STATS), ("full_check", int(full_check))],
        _taa(stats),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, consumer], semaphores=[], cbs=cbs)


def _overlap_timings(device, sd, run0, run1, reps):
    sync = lambda: ttnn.synchronize_device(device)

    def concurrent():
        run0()
        device.set_sub_device_stall_group([sd[1]])  # host work after this does not stall on sub-device 0
        run1()
        device.reset_sub_device_stall_group()

    def sequential():
        run0()
        sync()
        run1()

    t0, _ = _timed(run0, sync, reps)
    t1, _ = _timed(run1, sync, reps)
    t_seq, _ = _timed(sequential, sync, reps)
    t_conc, conc_all = _timed(concurrent, sync, reps)
    return {
        "alone_sd0_ms": t0 * 1e3,
        "alone_sd1_ms": t1 * 1e3,
        "sum_alone_ms": (t0 + t1) * 1e3,
        "sequential_ms": t_seq * 1e3,
        "concurrent_ms": t_conc * 1e3,
        "concurrent_runs_ms": [x * 1e3 for x in conc_all],
        "overlap_ratio": t_seq / t_conc,
        "ideal_ratio": t_seq / max(t0, t1),
    }


@pytest.mark.parametrize("iters", [131072])
def test_a_overlap_compute_bound(device, request, iters):
    left, right = _halves(device)
    torch.manual_seed(0)
    a = torch.randint(0, 3, (32, 32)).float()
    b = torch.randint(0, 3, (32, 32)).float()
    expected = (a @ b).to(torch.bfloat16)  # small integers: exact in bf16
    ab = ttnn.from_torch(
        torch.cat([a, b], dim=1).reshape(1, 1, 32, 64).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    hz = _wall_clock_hz(device)
    with _sub_devices(device, [left, right]) as sd:
        bufs = []
        for cores in (left, right):
            n = cores.num_cores()
            out = ttnn.from_torch(
                torch.zeros(1, 1, 32, 32 * n, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            stats = _u32_zeros(device, n, STATS_WORDS)
            bufs.append((cores, out, stats, _compute_program(cores, ab, out, stats, iters)))
        runs = [lambda o=o, s=s, p=p: ttnn.generic_op([ab, o, s], p) for _, o, s, p in bufs]

        runs[0]()
        runs[1]()
        ttnn.synchronize_device(device)
        timings = _overlap_timings(device, sd, runs[0], runs[1], reps=5)

        # The last concurrent run's per-core device intervals show the overlap directly.
        intervals, ref = [], None
        for cores, out, stats, _ in bufs:
            got = ttnn.to_torch(out).reshape(32, cores.num_cores(), 32).permute(1, 0, 2)
            for i in range(cores.num_cores()):
                assert torch.equal(got[i], expected), f"core {i} of {_desc(cores)}: wrong A@B"
            s = _u32_read(stats)
            ref = int(s[0, 0]) if ref is None else ref
            intervals.append((int(_ticks(s, 0, ref).min()), int(_ticks(s, 2, ref).max())))

    # The same per-core work on the whole grid without sub-devices: what the chip does under a full-chip
    # load by itself, to compare with the concurrent run.
    g = device.compute_with_storage_grid_size()
    full = _crs(0, 0, g.x - 1, g.y - 1)
    nf = full.num_cores()
    out_f = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32 * nf, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    stats_f = _u32_zeros(device, nf, STATS_WORDS)
    prog_f = _compute_program(full, ab, out_f, stats_f, iters)
    run_f = lambda: ttnn.generic_op([ab, out_f, stats_f], prog_f)
    run_f()
    ttnn.synchronize_device(device)
    t_full = _timed(run_f, lambda: ttnn.synchronize_device(device), 5)[0]
    got = ttnn.to_torch(out_f).reshape(32, nf, 32).permute(1, 0, 2)
    assert all(torch.equal(got[i], expected) for i in range(nf)), "full-grid run: wrong A@B"
    sf = _u32_read(stats_f)
    full_cycles = int(_ticks(sf, 2, int(sf[0, 0])).max() - _ticks(sf, 0, int(sf[0, 0])).min())

    flops = 2 * 32**3 * 4 * iters
    timings.update(
        {
            "full_grid_same_per_core_work_ms": t_full * 1e3,
            "full_grid_effective_clock_mhz": full_cycles / t_full / 1e6,
            "device_union_ms_last_concurrent": (max(e for _, e in intervals) - min(s for s, _ in intervals)) / hz * 1e3,
            "grid": [device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y],
            "sd0": _desc(left),
            "sd1": _desc(right),
            "iters": iters,
            "tile_matmuls_per_core": 4 * iters,
            "device_interval_overlap": _interval_overlap(*intervals),
            "device_busy_ms": [(e - s) / hz * 1e3 for s, e in intervals],
            "tflops_per_core": flops / ((intervals[0][1] - intervals[0][0]) / hz) / 1e12,
            "wall_clock_mhz": hz / 1e6,
        }
    )
    _record(request, timings)
    assert timings["overlap_ratio"] > MIN_COMPUTE_OVERLAP, (
        f"compute-bound overlap {timings['overlap_ratio']:.2f} <= {MIN_COMPUTE_OVERLAP}: the two sub-devices did "
        f"not run concurrently (device interval overlap {timings['device_interval_overlap']:.2f})"
    )


@pytest.mark.parametrize("page_size, total_mb, passes", [(4096, 64, 64)])
def test_a_overlap_dram_bound(device, request, page_size, total_mb, passes):
    left, right = _halves(device)
    hz = _wall_clock_hz(device)
    words = page_size // 4
    rng = np.random.default_rng(1)
    with _sub_devices(device, [left, right]) as sd:
        bufs = []
        for cores in (left, right):
            n = cores.num_cores()
            # Equal bytes per sub-device, so a perfect overlap would give the full 2x.
            pages_per_core = (total_mb << 20) // page_size // n // 8 * 8
            data = rng.integers(0, 2**32, size=(pages_per_core * n, words), dtype=np.uint64)
            src = _u32_tensor(device, data)
            stats = _u32_zeros(device, n, STATS_WORDS)
            per_core = data.reshape(n, pages_per_core, words)
            bufs.append(
                {
                    "word0_col": data[:, 0].copy(),
                    "cores": cores,
                    "src": src,
                    "stats": stats,
                    "pages_per_core": pages_per_core,
                    "full": _dram_program(cores, src, stats, pages_per_core, 1, True, page_size),
                    "timed": _dram_program(cores, src, stats, pages_per_core, passes, False, page_size),
                    "sum": per_core.sum(axis=(1, 2)) & 0xFFFFFFFF,
                    "wsum": (per_core * np.arange(1, words + 1, dtype=np.uint64)).sum(axis=(1, 2)) & 0xFFFFFFFF,
                    "word0": (per_core[:, :, 0].sum(axis=1) * passes) & 0xFFFFFFFF,
                    "bytes": pages_per_core * n * page_size,
                }
            )

        # Byte-level check of what each core streams (every word, position weighted), both sub-devices at once.
        for b in bufs:
            ttnn.generic_op([b["src"], b["stats"]], b["full"])
        ttnn.synchronize_device(device)
        for b in bufs:
            s = _u32_read(b["stats"])
            assert np.array_equal(s[:, 4], b["sum"]) and np.array_equal(s[:, 5], b["wsum"]), _desc(b["cores"])
            assert np.all(s[:, 6] == b["pages_per_core"])

        runs = [lambda b=b: ttnn.generic_op([b["src"], b["stats"]], b["timed"]) for b in bufs]
        runs[0]()
        runs[1]()
        ttnn.synchronize_device(device)
        timings = _overlap_timings(device, sd, runs[0], runs[1], reps=5)
        intervals, ref = [], None
        for b in bufs:
            s = _u32_read(b["stats"])
            assert np.array_equal(s[:, 4], b["word0"]), f"{_desc(b['cores'])}: stream checksum mismatch"
            assert np.all(s[:, 6] == b["pages_per_core"] * passes)
            ref = int(s[0, 0]) if ref is None else ref
            intervals.append((int(_ticks(s, 0, ref).min()), int(_ticks(s, 2, ref).max())))
    # The same bytes as sub-device 1 streams, on the full grid without sub-devices: is DRAM the cap?
    g = device.compute_with_storage_grid_size()
    full = _crs(0, 0, g.x - 1, g.y - 1)
    b = bufs[1]
    ppc = len(b["word0_col"]) // full.num_cores() // 8 * 8
    stats_f = _u32_zeros(device, full.num_cores(), STATS_WORDS)
    prog_f = _dram_program(full, b["src"], stats_f, ppc, passes, False, page_size)
    run_f = lambda: ttnn.generic_op([b["src"], stats_f], prog_f)
    run_f()
    ttnn.synchronize_device(device)
    t_full = _timed(run_f, lambda: ttnn.synchronize_device(device), 5)[0]
    s = _u32_read(stats_f)
    exp = (b["word0_col"][: ppc * full.num_cores()].reshape(full.num_cores(), ppc).sum(axis=1) * passes) & 0xFFFFFFFF
    assert np.array_equal(s[:, 4], exp), "full-grid stream checksum mismatch"
    timings["full_grid_gbps"] = ppc * full.num_cores() * page_size * passes / t_full / 1e9
    gb = [b["bytes"] * passes / 1e9 for b in bufs]
    timings.update(
        {
            "sd0": _desc(left),
            "sd1": _desc(right),
            "page_size": page_size,
            "bytes_per_sd_per_run": [b["bytes"] * passes for b in bufs],
            "alone_gbps": [gb[0] / (timings["alone_sd0_ms"] / 1e3), gb[1] / (timings["alone_sd1_ms"] / 1e3)],
            "concurrent_total_gbps": sum(gb) / (timings["concurrent_ms"] / 1e3),
            "device_interval_overlap": _interval_overlap(*intervals),
            "device_busy_ms_last_concurrent": [(e - s) / hz * 1e3 for s, e in intervals],
            "wall_clock_mhz": hz / 1e6,
        }
    )
    _record(request, timings)


# ----------------------------------------------------------------------------------------------------------
# (b) Global-semaphore handoff
# ----------------------------------------------------------------------------------------------------------


def _expected_payload(i, seed):
    return (i * 0x9E3779B1 + seed) & 0xFFFFFFFF, ((i ^ seed) + 0x7F4A7C15) & 0xFFFFFFFF


@pytest.mark.parametrize("with_payload", [True, False], ids=["payload", "signal_only"])
@pytest.mark.parametrize("pair", ["near", "far"])
def test_b_global_semaphore_handoff(device, request, pair, with_payload):
    n_iter = 1000
    left, right = _halves(device)
    g = device.compute_with_storage_grid_size()
    split = g.x // 2
    prod, cons = {
        "near": (ttnn.CoreCoord(split - 1, 0), ttnn.CoreCoord(split, 0)),
        "far": (ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1)),
    }[pair]
    pv, cv = device.worker_core_from_logical_core(prod), device.worker_core_from_logical_core(cons)
    both = ttnn.CoreRangeSet({ttnn.CoreRange(prod, prod), ttnn.CoreRange(cons, cons)})
    hz = _wall_clock_hz(device)
    with _sub_devices(device, [left, right]) as sd:
        ready, ack, mbox0, mbox1 = [ttnn.create_global_semaphore(device, both, 0) for _ in range(4)]
        addr = [ttnn.get_global_semaphore_address(s) for s in (ready, ack, mbox0, mbox1)]
        log = _u32_zeros(device, 1, 4 * n_iter)
        pstats = _u32_zeros(device, 1, STATS_WORDS)
        prod_crs, cons_crs = _crs(prod.x, prod.y, prod.x, prod.y), _crs(cons.x, cons.y, cons.x, cons.y)
        cons_prog = ttnn.ProgramDescriptor(
            kernels=[
                _kernel(
                    "sem_consumer.cpp",
                    cons_crs,
                    ttnn.WriterConfigDescriptor(),
                    [(cons, [pv.x, pv.y, *addr, n_iter, log.buffer_address()])],
                    [("cb_log", 0)],
                    _taa(log),
                )
            ],
            semaphores=[],
            cbs=[_cb(0, cons_crs, 16 * n_iter, 1, ttnn.uint32)],
        )
        results = []
        for seed in (0x1234, 0xBEEF, 0x5A5A5A5A):
            for s in (ready, ack, mbox0, mbox1):
                ttnn.reset_global_semaphore_value(s, 0)
            ttnn.synchronize_device(device)
            prod_prog = ttnn.ProgramDescriptor(
                kernels=[
                    _kernel(
                        "sem_producer.cpp",
                        prod_crs,
                        ttnn.WriterConfigDescriptor(),
                        [(prod, [cv.x, cv.y, *addr, n_iter, seed, pstats.buffer_address()])],
                        [("with_payload", int(with_payload)), ("cb_stats", CB_STATS)],
                        _taa(pstats),
                    )
                ],
                semaphores=[],
                cbs=[_cb(CB_STATS, prod_crs, 64, 1, ttnn.uint32)],
            )
            # The consumer spins on sub-device 1 until the producer on sub-device 0 runs: this completes only
            # because the two sub-devices dispatch independently.
            _launch_pair(
                device,
                sd,
                lambda: ttnn.generic_op([log, log], cons_prog),
                lambda: ttnn.generic_op([pstats, pstats], prod_prog),
            )

            got = _u32_read(log).reshape(n_iter, 4)
            idx = np.arange(1, n_iter + 1)
            assert np.array_equal(got[:, 0], idx), "consumer saw the ready semaphore out of order"
            if with_payload:
                exp = np.array([_expected_payload(int(i), seed) for i in idx])
                assert np.array_equal(got[:, 1:3], exp), "payload does not match its iteration"
            s = _u32_read(pstats)[0]
            assert s[8] == n_iter and s[9] == n_iter
            results.append(
                {
                    "seed": seed,
                    "rtt_min_ns": s[4] / hz * 1e9,
                    "rtt_max_ns": s[5] / hz * 1e9,
                    "rtt_mean_ns": ((s[7] << 32) | s[6]) / n_iter / hz * 1e9,
                    "rtt_mean_ticks": ((s[7] << 32) | s[6]) / n_iter,
                }
            )
        # For comparison, a host-mediated handoff: a trivial program on the producer core completes, the host
        # sees it, then launches one on the consumer core (two launch + wait round trips per measurement).
        pspin, cspin = _u32_zeros(device, 1, STATS_WORDS), _u32_zeros(device, 1, STATS_WORDS)
        run_p = lambda p=_spin_program(prod, pspin, 1): ttnn.generic_op([pspin, pspin], p)
        run_c = lambda p=_spin_program(cons, cspin, 1): ttnn.generic_op([cspin, cspin], p)
        sync = lambda: ttnn.synchronize_device(device)
        run_p(), sync(), run_c(), sync()
        host_rt = _timed(lambda: (run_p(), sync(), run_c()), sync, 50)[0] / 2
        del ready, ack, mbox0, mbox1
    data = {
        "pair": pair,
        "producer": [prod.x, prod.y],
        "consumer": [cons.x, cons.y],
        "with_payload": with_payload,
        "iterations": n_iter,
        "runs": results,
        "rtt_mean_ns": statistics.median(r["rtt_mean_ns"] for r in results),
        "rtt_min_ns": min(r["rtt_min_ns"] for r in results),
        "handoff_one_way_ns": statistics.median(r["rtt_mean_ns"] for r in results) / 2,
        "host_mediated_handoff_us": host_rt * 1e6,
        "wall_clock_mhz": hz / 1e6,
    }
    _record(request, data)


# ----------------------------------------------------------------------------------------------------------
# (c) Global circular buffer streaming
# ----------------------------------------------------------------------------------------------------------


def _gcb_programs(device, gcb, senders, receivers, src, out, sstats, rstats, num_pages, page_size, mode, ring):
    s_crs = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in senders})
    r_crs = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in receivers})
    src_pages = num_pages if mode == 0 else ring
    sender_cbs = [
        _cb(0, s_crs, page_size, 4 if mode == 0 else ring, ttnn.uint32),
        _remote_cb(31, s_crs, page_size, gcb),
        _cb(CB_STATS, s_crs, 64, 1, ttnn.uint32),
    ]
    reader = _kernel(
        "reader_dram_stream.cpp",
        s_crs,
        ttnn.ReaderConfigDescriptor(),
        [(c, [src.buffer_address(), k * src_pages, src_pages, 1]) for k, c in enumerate(senders)],
        [("cb_data", 0), ("batch", 1 if mode == 0 else ring), ("page_size", page_size)],
        _taa(src),
    )
    named = [("cb_remote", 31), ("cb_stats", CB_STATS), ("page_size", page_size), ("mode", mode), ("ring", ring)]
    sender = _kernel(
        "gcb_sender.cpp",
        s_crs,
        ttnn.WriterConfigDescriptor(),
        [(c, [num_pages, sstats.buffer_address(), k]) for k, c in enumerate(senders)],
        named + [("cb_src", 0)],
        _taa(sstats),
    )
    receiver = _kernel(
        "gcb_receiver.cpp",
        r_crs,
        ttnn.WriterConfigDescriptor(),
        [
            (
                c,
                [
                    num_pages,
                    out.buffer_address(),
                    k * (num_pages if mode == 0 else ring),
                    rstats.buffer_address(),
                    k,
                    k * ring,
                ],
            )
            for k, c in enumerate(receivers)
        ],
        named,
        _taa(out) + _taa(rstats),
    )
    send = ttnn.ProgramDescriptor(kernels=[reader, sender], semaphores=[], cbs=sender_cbs)
    recv = ttnn.ProgramDescriptor(
        kernels=[receiver],
        semaphores=[],
        cbs=[_remote_cb(31, r_crs, page_size, gcb), _cb(CB_STATS, r_crs, 64, 1, ttnn.uint32)],
    )
    return send, recv


@pytest.mark.parametrize("page_size", [2048, 8192])
@pytest.mark.parametrize("num_pairs", [1, 10])
def test_c_global_cb_streaming(device, request, num_pairs, page_size):
    depth = 4
    ring = 2 * depth  # > depth, so a sender that overran the receiver would show up as a tag mismatch
    check_pages, bw_pages = 256, 16384 if page_size == 2048 else 4096
    left, right = _halves(device)
    g = device.compute_with_storage_grid_size()
    split = g.x // 2
    num_pairs = min(num_pairs, g.y)
    # Sender in the last column of sub-device 0, receiver in the first column of sub-device 1, one pair per row.
    senders = [ttnn.CoreCoord(split - 1, y) for y in range(num_pairs)]
    receivers = [ttnn.CoreCoord(split, y) for y in range(num_pairs)]
    words = page_size // 4
    hz = _wall_clock_hz(device)
    rng = np.random.default_rng(2)
    with _sub_devices(device, [left, right]) as sd:
        gcb = ttnn.create_global_circular_buffer(
            device, [(s, _crs(r.x, r.y, r.x, r.y)) for s, r in zip(senders, receivers)], depth * page_size
        )
        sstats = _u32_zeros(device, num_pairs, STATS_WORDS)
        rstats = _u32_zeros(device, num_pairs, STATS_WORDS)

        # 1) Byte-exact: DRAM -> sender L1 -> global CB -> receiver L1 -> DRAM, check_pages per pair.
        data = rng.integers(0, 2**32, size=(num_pairs * check_pages, words), dtype=np.uint64)
        src = _u32_tensor(device, data)
        out = _u32_zeros(device, num_pairs * check_pages, words)
        send, recv = _gcb_programs(
            device, gcb, senders, receivers, src, out, sstats, rstats, check_pages, page_size, 0, ring
        )
        for run in range(2):  # the second run starts from the fifo pointers the first one committed
            if run:
                data = rng.integers(0, 2**32, size=data.shape, dtype=np.uint64)
                host = torch.from_numpy(data.astype(np.uint32).view(np.int32))
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=ttnn.uint32), src)
            _launch_pair(
                device, sd, lambda: ttnn.generic_op([out, rstats], recv), lambda: ttnn.generic_op([src, sstats], send)
            )
            assert np.array_equal(_u32_read(out), data & 0xFFFFFFFF), "global CB stream is not byte-exact"

        # 2) Bandwidth: senders push `ring` preloaded pages round-robin (L1 -> L1 only); receivers check tags.
        tagged = rng.integers(0, 2**32, size=(num_pairs * ring, words), dtype=np.uint64)
        tagged[:, 0] = np.arange(num_pairs * ring)
        src_bw = _u32_tensor(device, tagged)
        send_bw, recv_bw = _gcb_programs(
            device, gcb, senders, receivers, src_bw, out, sstats, rstats, bw_pages, page_size, 1, ring
        )
        runs = []
        for _ in range(4):
            t0 = time.perf_counter()
            _launch_pair(
                device,
                sd,
                lambda: ttnn.generic_op([out, rstats], recv_bw),
                lambda: ttnn.generic_op([src_bw, sstats], send_bw),
            )
            host_s = time.perf_counter() - t0
            ss, rs = _u32_read(sstats), _u32_read(rstats)
            assert np.all(rs[:, 4] == bw_pages) and np.all(rs[:, 5] == 0), "receiver saw missing or overrun pages"
            assert np.all(ss[:, 4] == bw_pages)
            last = _u32_read(out)[: num_pairs * ring]
            assert np.array_equal(last, tagged & 0xFFFFFFFF), "last pages through the global CB are not byte-exact"
            # Receiver side: first page arrival to last pop; sender side: first push to last ack.
            ref = int(ss[0, 0])
            r_first, r_end = _ticks(rs, 6, ref), _ticks(rs, 2, ref)
            s_start, s_end = _ticks(ss, 0, ref), _ticks(ss, 2, ref)
            moved = (bw_pages - 1) * page_size
            dev_s = (int(r_end.max()) - int(r_first.min())) / hz
            runs.append(
                {
                    "host_ms": host_s * 1e3,
                    "receiver_ms": dev_s * 1e3,
                    "sender_ms": (int(s_end.max()) - int(s_start.min())) / hz * 1e3,
                    "aggregate_gbps": num_pairs * moved / dev_s / 1e9,
                    "per_pair_gbps_mean": float((moved / ((r_end - r_first) / hz)).mean() / 1e9),
                    "host_aggregate_gbps": num_pairs * bw_pages * page_size / host_s / 1e9,
                }
            )
    best = runs[1:]  # the first run includes kernel compile / first launch
    data = {
        "num_pairs": num_pairs,
        "page_size": page_size,
        "gcb_bytes_per_receiver": depth * page_size,
        "byte_exact_pages_per_pair": check_pages,
        "bandwidth_pages_per_pair": bw_pages,
        "runs": runs,
        "aggregate_gbps": statistics.median(r["aggregate_gbps"] for r in best),
        "per_pair_gbps": statistics.median(r["per_pair_gbps_mean"] for r in best),
        "wall_clock_mhz": hz / 1e6,
    }
    _record(request, data)


# ----------------------------------------------------------------------------------------------------------
# (d) Co-hosted workloads: partitioned + concurrent vs time-sliced on the full grid
# ----------------------------------------------------------------------------------------------------------


def _speech_config(cores, m_tiles, n_tiles):
    """2D-mcast config for an M x N output spread over the bounding box of `cores`."""
    grid = list(cores.ranges())[0].grid_size()
    per_m = -(-m_tiles // grid.y)
    per_n = -(-n_tiles // grid.x)
    sub_w = max(d for d in range(1, 9) if per_n % d == 0)
    sub_h = max(d for d in range(1, 9) if per_m % d == 0 and d * sub_w <= 8) if sub_w == per_n else 1
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        in0_block_w=4,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=per_m,
        per_core_N=per_n,
        transpose_mcast=False,
        fused_activation=None,
        allowed_worker_cores=cores,
    )


def _decode_config(cores, n_tiles):
    """1D-mcast (in0 multicast, N split across cores) config for a 32-row decode matmul on `cores`."""
    grid = list(cores.ranges())[0].grid_size()
    per_n = -(-n_tiles // cores.num_cores())
    sub_w = max(d for d in range(1, 9) if per_n % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=sub_w,
        per_core_M=1,
        per_core_N=per_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
        allowed_worker_cores=cores,
    )


def _trace_time(device, body, reps):
    """Capture body() once as a trace, then return the median replay time (s) over reps replays."""
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    keep = body()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    t, _ = _timed(
        lambda: ttnn.execute_trace(device, tid, cq_id=0, blocking=False), lambda: ttnn.synchronize_device(device), reps
    )
    ttnn.release_trace(device, tid)
    del keep
    return t


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 << 20}], indirect=True)
@pytest.mark.parametrize("split", [4, None, 8], ids=["speech4cols", "half", "speech8cols"])
@pytest.mark.parametrize("speech_m", [1024, 4096])
@pytest.mark.parametrize("n", [32])
def test_d_cohosted_matmuls(device, request, n, speech_m, split):
    reps = 5
    left, right = _halves(device, split)
    torch.manual_seed(3)
    dram = ttnn.DRAM_MEMORY_CONFIG
    xs, ws = torch.randn(1, 1, speech_m, 1024), torch.randn(1, 1, 1024, 1024) / 32
    xd, wd = torch.randn(1, 1, 32, 4096), torch.randn(1, 1, 4096, 14336) / 64
    ref_s, ref_d = xs @ ws, xd @ wd
    xs_t = ttnn.from_torch(xs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    ws_t = ttnn.from_torch(ws, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    xd_t = ttnn.from_torch(xd, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    wd_t = ttnn.from_torch(wd, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    hifi2 = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2)
    lofi = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.LoFi)

    def speech(cfg=None):
        return ttnn.matmul(xs_t, ws_t, program_config=cfg, memory_config=dram, compute_kernel_config=hifi2)

    def decode(cfg=None):
        return ttnn.matmul(xd_t, wd_t, program_config=cfg, memory_config=dram, compute_kernel_config=lofi)

    sync = lambda: ttnn.synchronize_device(device)
    res = {
        "speech_shape_mkn": [speech_m, 1024, 1024],
        "decode_shape_mkn": [32, 4096, 14336],
        "sd0_speech": _desc(left),
        "sd1_decode": _desc(right),
        "n_per_workload": n,
        "reps": reps,
    }

    # Full grid, no sub-devices: each alone, and both time-sliced (today's co-hosting).
    s_full, d_full = speech(), decode()
    sync()
    assert_with_pcc(ref_s, ttnn.to_torch(s_full), 0.999)
    assert_with_pcc(ref_d, ttnn.to_torch(d_full), 0.99)
    eager = {}
    eager["speech_full_ms"] = _timed(lambda: [speech() for _ in range(n)], sync, reps)[0] * 1e3 / n
    eager["decode_full_ms"] = _timed(lambda: [decode() for _ in range(n)], sync, reps)[0] * 1e3 / n
    eager["timesliced_full_ms"] = _timed(lambda: [(speech(), decode()) for _ in range(n)], sync, reps)[0] * 1e3 / n
    traced = {
        "speech_full_ms": _trace_time(device, lambda: [speech() for _ in range(n)], reps) * 1e3 / n,
        "decode_full_ms": _trace_time(device, lambda: [decode() for _ in range(n)], reps) * 1e3 / n,
        "timesliced_full_ms": _trace_time(device, lambda: [(speech(), decode()) for _ in range(n)], reps) * 1e3 / n,
    }

    # Time-sliced again, with the program configs the sub-grids use but sized for the whole grid, so that a
    # partition gain is not just a better config than the default one.
    g = device.compute_with_storage_grid_size()
    full = _crs(0, 0, g.x - 1, g.y - 1)
    cfg_sf, cfg_df = _speech_config(full, speech_m // 32, 32), _decode_config(full, 14336 // 32)
    res["speech_config_full"], res["decode_config_full"] = repr(cfg_sf), repr(cfg_df)
    assert_with_pcc(ref_s, ttnn.to_torch(speech(cfg_sf)), 0.999)
    assert_with_pcc(ref_d, ttnn.to_torch(decode(cfg_df)), 0.99)
    for t, timer in (
        (eager, lambda body: _timed(body, sync, reps)[0]),
        (traced, lambda body: _trace_time(device, body, reps)),
    ):
        t["speech_full_cfg_ms"] = timer(lambda: [speech(cfg_sf) for _ in range(n)]) * 1e3 / n
        t["decode_full_cfg_ms"] = timer(lambda: [decode(cfg_df) for _ in range(n)]) * 1e3 / n
        t["timesliced_full_cfg_ms"] = timer(lambda: [(speech(cfg_sf), decode(cfg_df)) for _ in range(n)]) * 1e3 / n

    cfg_s, cfg_d = _speech_config(left, speech_m // 32, 32), _decode_config(right, 14336 // 32)
    res["speech_config"], res["decode_config"] = repr(cfg_s), repr(cfg_d)
    with _sub_devices(device, [left, right]):
        s_sub, d_sub = speech(cfg_s), decode(cfg_d)
        sync()
        s_sub_host, d_sub_host = ttnn.to_torch(s_sub), ttnn.to_torch(d_sub)
        assert_with_pcc(ref_s, s_sub_host, 0.999)
        assert_with_pcc(ref_d, d_sub_host, 0.99)
        # Interleaved enqueue across the two sub-devices must give the same bits as running alone.
        pairs = [(speech(cfg_s), decode(cfg_d)) for _ in range(4)]
        sync()
        for s_t, d_t in pairs:
            assert torch.equal(ttnn.to_torch(s_t), s_sub_host) and torch.equal(ttnn.to_torch(d_t), d_sub_host)
        del pairs

        eager["speech_sub_ms"] = _timed(lambda: [speech(cfg_s) for _ in range(n)], sync, reps)[0] * 1e3 / n
        eager["decode_sub_ms"] = _timed(lambda: [decode(cfg_d) for _ in range(n)], sync, reps)[0] * 1e3 / n
        eager["concurrent_sub_ms"] = (
            _timed(lambda: [(speech(cfg_s), decode(cfg_d)) for _ in range(n)], sync, reps)[0] * 1e3 / n
        )
        traced["speech_sub_ms"] = _trace_time(device, lambda: [speech(cfg_s) for _ in range(n)], reps) * 1e3 / n
        traced["decode_sub_ms"] = _trace_time(device, lambda: [decode(cfg_d) for _ in range(n)], reps) * 1e3 / n
        traced["concurrent_sub_ms"] = (
            _trace_time(device, lambda: [(speech(cfg_s), decode(cfg_d)) for _ in range(n)], reps) * 1e3 / n
        )

    for name, t in (("eager", eager), ("traced", traced)):
        t["speedup_partitioned_vs_timesliced"] = t["timesliced_full_ms"] / t["concurrent_sub_ms"]
        t["best_timesliced_full_ms"] = min(t["timesliced_full_ms"], t["timesliced_full_cfg_ms"])
        t["speedup_partitioned_vs_best_timesliced"] = t["best_timesliced_full_ms"] / t["concurrent_sub_ms"]
        t["sum_alone_full_ms"] = t["speech_full_ms"] + t["decode_full_ms"]
        # Perfect overlap on the sub-grids would take as long as the slower of the two alone.
        t["ideal_concurrent_sub_ms"] = max(t["speech_sub_ms"], t["decode_sub_ms"])
        t["ideal_speedup"] = t["timesliced_full_ms"] / t["ideal_concurrent_sub_ms"]
        res[name] = t
    weight_bytes = (4096 // 32) * (14336 // 32) * 1088  # bfloat8_b tiles
    res["decode_weight_gbps_traced"] = {
        k: weight_bytes / (traced[k] / 1e3) / 1e9 for k in ("decode_full_ms", "decode_sub_ms")
    }
    _record(request, res)


# ----------------------------------------------------------------------------------------------------------
# What an unmodified op does under a partition
# ----------------------------------------------------------------------------------------------------------


def test_e_default_grid_op_rejected_while_partitioned(device, request, expect_error):
    """An op left on its default (full) grid spans both sub-devices, and the runtime refuses to enqueue it:
    every op of a co-hosted model has to be given its sub-grid explicitly."""
    left, right = _halves(device)
    torch.manual_seed(4)
    xt = torch.randn(1, 1, 256, 256)
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    assert_with_pcc(xt @ xt, ttnn.to_torch(ttnn.matmul(x, x)), 0.999)
    message = "Programs must be executed on a single sub-device"
    with _sub_devices(device, [left, right]):
        with expect_error(RuntimeError, message) as err:
            ttnn.matmul(x, x)
            ttnn.synchronize_device(device)
        # The same op pinned to sub-device 0 runs fine.
        assert_with_pcc(xt @ xt, ttnn.to_torch(ttnn.matmul(x, x, program_config=_speech_config(left, 8, 8))), 0.999)
    _record(request, {"default_grid_op_error": next(l for l in str(err.value).splitlines() if message in l).strip()})
