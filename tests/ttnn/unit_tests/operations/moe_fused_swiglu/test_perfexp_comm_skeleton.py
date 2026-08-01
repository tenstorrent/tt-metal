# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Thin pytest entry point for the comm_skeleton cost-model bench.

ALL logic (program descriptors, kernels, probes) lives under the experiment dir:
    ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/comm_skeleton/

This file only exists here (rather than inside that dir) because pytest's --import-mode=importlib
cannot collect a test_*.py placed directly inside the `ttnn/ttnn/` package tree: it derives a dotted
module path starting with "ttnn" and ends up re-executing ttnn/ttnn/__init__.py under a second
qualified name, which crashes on duplicate C++ op registration. Named uniquely for this idea so it
cannot collide with a sibling part-optimizer's entry point.

    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_perfexp_comm_skeleton.py::test_smoke
    ... ::test_launch_intercept
    ... ::test_cb_cost
    ... ::test_noc_issue_cost
    ... ::test_noc_payload_cost
    ... ::test_sem_cost
    ... ::test_mcast_rendezvous
    ... ::test_dest_sync_cost
"""

import torch
import pytest
from loguru import logger

import ttnn
from ttnn.operations.moe_fused_swiglu.perf_experiments.comm_skeleton import comm_bench as cb


# The op's real worker grid is 11x8 (88 cores); 11x10 (110) is this box's full grid; 1x1 is the
# single-core control that shows what is core-count-independent.
GRIDS = [(1, 1), (11, 8), (11, 10)]


def _grids(device):
    out = []
    for h, k in GRIDS:
        hh, kk = cb.clamp_grid(device, h, k)
        if (hh, kk) not in out:
            out.append((hh, kk))
    return out


def _report(title, rows):
    lines = ["", f"=== {title} ==="] + list(rows)
    logger.info("\n".join(lines))


def _sweep(device, run_fn, xs):
    """One fresh run per point (no averaging loop — device kernel time has no warm-up transient)."""
    return [cb.measure_once(device, lambda n=n: run_fn(n)) for n in xs]


def _fit_line(name, xs, ys, unit="op"):
    slope, intercept, r2 = cb.fit(xs, ys)
    pts = "  ".join(f"{n}:{y:.0f}" for n, y in zip(xs, ys))
    return f"  {name:<34} slope {slope:8.2f} ns/{unit}   intercept {intercept:8.0f} ns   r2 {r2:.4f}   [{pts}]"


# ---------------------------------------------------------------------------
# smoke — every probe runs once at a small size. Correctness here means "completes, no hang, and
# (for the multicast) the payload actually landed". Run this before any sweep.
# ---------------------------------------------------------------------------
def test_smoke(device):
    gx, gy = cb.device_grid(device)
    logger.info(f"device compute grid = {gx}x{gy} ({gx*gy} cores)")

    cb.run_launch(device, 1, 1)
    cb.run_cb(device, cb.CB_MODE_PROD_DEEP, n_iters=8, pages_per_call=1, hgroups=1, kgroups=1)
    cb.run_cb(device, cb.CB_MODE_PAIR_SHALLOW, n_iters=8, pages_per_call=1, hgroups=1, kgroups=1, depth_pages=2)
    cb.run_cb(device, cb.CB_MODE_CONS_PREFILL, n_iters=8, pages_per_call=1, hgroups=1, kgroups=1)

    scratch = cb.make_scratch(device)
    for mode in sorted(cb.NOC_MODES):
        cb.run_noc(
            device,
            mode,
            n_cmds=8,
            xfer_bytes=32,
            barrier_each=False,
            on_writer=False,
            hgroups=2,
            kgroups=2,
            scratch=scratch,
        )

    cb.run_sem_inc_p2p(device, n_ops=8, barrier_each=True, on_writer=True, hgroups=2, kgroups=2)
    cb.run_sem_incast(device, n_rounds=4, n_senders=4, hgroups=2, kgroups=2)
    cb.run_sem_wait_sat(device, n_ops=8, hgroups=1, kgroups=1)
    cb.run_sem_mcast(device, n_ops=8, barrier_each=True, hgroups=2, kgroups=2)

    cb.run_dest(device, n_iters=8, hgroups=1, kgroups=1)

    # multicast rendezvous + its correctness gate
    h, k = cb.clamp_grid(device, 11, 8)
    _mcast_verify(device, rounds=h, stage=cb.MCAST_STAGE_FULL, depth=1, payload_bytes=64, hgroups=h, kgroups=k)
    logger.info("smoke: all probes completed")


def _mcast_verify(device, *, rounds, stage, depth, payload_bytes, hgroups, kgroups):
    """CORRECTNESS GATE for the rendezvous: every sender broadcasts the SAME source page, so after
    the run every core's slot must hold exactly those bytes. Proves the multicast landed."""
    num_cores = hgroups * kgroups
    io = cb.make_mcast_io(device, payload_bytes, num_cores)
    src, _, tt_out = io
    cb.run_mcast(
        device,
        rounds=rounds,
        stage=stage,
        depth=depth,
        payload_bytes=payload_bytes,
        hgroups=hgroups,
        kgroups=kgroups,
        verify=True,
        io=io,
    )
    # via int64: torch refuses to promote uint32 against int32, and the payload is opaque bytes.
    got = ttnn.to_torch(tt_out).to(torch.int64)
    expected = src.repeat(num_cores, 1).to(torch.int64)
    bad = (got != expected).any(dim=1).nonzero().flatten().tolist()
    assert not bad, f"multicast payload did not land on core indices {bad[:8]}{'...' if len(bad) > 8 else ''}"


def test_mcast_correctness(device):
    """The rendezvous's payload gate at both the op's grid and the full grid, serial and pipelined."""
    for h, k in _grids(device):
        if h * k < 4:
            continue
        for depth in (1, 4):
            _mcast_verify(
                device, rounds=2 * h, stage=cb.MCAST_STAGE_FULL, depth=depth, payload_bytes=64, hgroups=h, kgroups=k
            )
    logger.info("mcast rendezvous: payload verified on every core, every grid, depth 1 and 4")


# ---------------------------------------------------------------------------
# PROBE 0 — the launch intercept
# ---------------------------------------------------------------------------
def test_launch_intercept(device):
    rows = []
    for h, k in _grids(device):
        ns_one = cb.measure_once(device, lambda: cb.run_launch(device, h, k, both_riscs=False))
        ns_two = cb.measure_once(device, lambda: cb.run_launch(device, h, k, both_riscs=True))
        rows.append(f"  grid {h}x{k} ({h*k:>3} cores)   1 RISC-V: {ns_one:8.0f} ns    2 RISC-V: {ns_two:8.0f} ns")
    _report("launch intercept (empty kernel)", rows)


# ---------------------------------------------------------------------------
# PROBE 1 — CB bookkeeping
# ---------------------------------------------------------------------------
CB_ITERS = (64, 128, 256, 512, 1024)


def test_cb_cost(device):
    rows = []
    for h, k in _grids(device):
        for mode in sorted(cb.CB_MODES):
            for ppc in (1, 2, 4):
                depth = 4 * ppc if mode == cb.CB_MODE_PAIR_SHALLOW else None
                xs = [n for n in CB_ITERS if (mode != cb.CB_MODE_PAIR_SHALLOW or (n * ppc) % (4 * ppc) == 0)]
                ys = _sweep(
                    device,
                    lambda n, m=mode, p=ppc, d=depth: cb.run_cb(
                        device, m, n_iters=n, pages_per_call=p, hgroups=h, kgroups=k, depth_pages=d
                    ),
                    xs,
                )
                rows.append(_fit_line(f"{h}x{k} {cb.CB_MODES[mode]} ppc={ppc}", xs, ys, unit="cycle"))
    _report("CB bookkeeping, no payload (slope = ns per reserve/push[/wait/pop] CYCLE)", rows)


def test_cb_granularity(device):
    """PerChunk vs per-tile: hold TOTAL PAGES fixed and vary pages-per-call. A per-CALL cost makes
    the time fall as 1/ppc; a per-PAGE cost leaves it flat."""
    h, k = cb.clamp_grid(device, 11, 8)
    total_pages = 1024
    rows = []
    for mode in (cb.CB_MODE_PROD_DEEP, cb.CB_MODE_PAIR_SHALLOW):
        for ppc in (1, 2, 4, 8, 16, 32):
            n = total_pages // ppc
            depth = 4 * ppc if mode == cb.CB_MODE_PAIR_SHALLOW else None
            ns = cb.measure_once(
                device,
                lambda n=n, p=ppc, m=mode, d=depth: cb.run_cb(
                    device, m, n_iters=n, pages_per_call=p, hgroups=h, kgroups=k, depth_pages=d
                ),
            )
            rows.append(f"  {cb.CB_MODES[mode]:<14} ppc={ppc:<3} calls={n:<5} {ns:9.0f} ns  ({ns/n:7.2f} ns/call)")
    _report(f"CB granularity at fixed {total_pages} total pages, grid {h}x{k}", rows)


# ---------------------------------------------------------------------------
# PROBE 2 — NoC command issue cost (THE headline number)
# ---------------------------------------------------------------------------
NOC_COUNTS = (32, 64, 128, 256, 512)


def test_noc_issue_cost(device):
    """COUNT sweep at 32 B: the slope is ns per ISSUED command."""
    scratch = cb.make_scratch(device)
    rows = []
    for h, k in _grids(device):
        for on_writer in (False, True):
            risc = "BRISC/noc1" if on_writer else "NCRISC/noc0"
            for mode in sorted(cb.NOC_MODES):
                for barrier_each in (False, True):
                    tag = "bar/cmd" if barrier_each else "1 barrier"
                    xs = list(NOC_COUNTS)
                    ys = _sweep(
                        device,
                        lambda n, m=mode, b=barrier_each, w=on_writer: cb.run_noc(
                            device,
                            m,
                            n_cmds=n,
                            xfer_bytes=32,
                            barrier_each=b,
                            on_writer=w,
                            hgroups=h,
                            kgroups=k,
                            scratch=scratch,
                        ),
                        xs,
                    )
                    rows.append(_fit_line(f"{h}x{k} {risc} {cb.NOC_MODES[mode]:<19} {tag}", xs, ys, unit="cmd"))
    _report("NoC command ISSUE cost @ 32 B payload (slope = ns per command)", rows)


def test_noc_focus(device):
    """FOCUS + repeat: the headline issue cost, twice, at 1 core and at the op's 88.

    Two sizes matter. 32 B is the pure ISSUE probe. 1088 B is a bfp8_b tile — the op's actual weight
    read — and the gap between them at 88 cores is DRAM BANDWIDTH, which is real work rather than
    bookkeeping. Reporting both is what keeps the two from being confused."""
    scratch = cb.make_scratch(device)
    rows = []
    for h, k in [(1, 1), cb.clamp_grid(device, 11, 8)]:
        for mode in (cb.NOC_L1_READ_REMOTE, cb.NOC_L1_WRITE_REMOTE, cb.NOC_DRAM_READ_ACC):
            for size in (32, 1088):
                per_cmd = []
                for _ in range(2):
                    lo = cb.measure_once(
                        device,
                        lambda: cb.run_noc(
                            device,
                            mode,
                            n_cmds=64,
                            xfer_bytes=size,
                            barrier_each=False,
                            on_writer=False,
                            hgroups=h,
                            kgroups=k,
                            scratch=scratch,
                        ),
                    )
                    hi = cb.measure_once(
                        device,
                        lambda: cb.run_noc(
                            device,
                            mode,
                            n_cmds=512,
                            xfer_bytes=size,
                            barrier_each=False,
                            on_writer=False,
                            hgroups=h,
                            kgroups=k,
                            scratch=scratch,
                        ),
                    )
                    per_cmd.append((hi - lo) / (512 - 64))
                spread = abs(per_cmd[0] - per_cmd[1]) / max(per_cmd) * 100
                rows.append(
                    f"  {h}x{k} {cb.NOC_MODES[mode]:<19} {size:>5} B   "
                    f"{per_cmd[0]:7.2f} / {per_cmd[1]:7.2f} ns/cmd   (spread {spread:4.1f}%)"
                )
    _report("NoC issue FOCUS: 2 samples, slope between N=64 and N=512, batched barrier", rows)


NOC_SIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def test_noc_payload_cost(device):
    """SIZE sweep at fixed count: slope = ns per byte (bandwidth); intercept/count = issue cost,
    cross-checking the count sweep from the other axis."""
    scratch = cb.make_scratch(device)
    n_cmds = 128
    rows = []
    for h, k in _grids(device):
        for mode in (cb.NOC_DRAM_READ_ACC, cb.NOC_DRAM_WRITE_ACC, cb.NOC_L1_READ_REMOTE):
            xs = list(NOC_SIZES)
            ys = _sweep(
                device,
                lambda s, m=mode: cb.run_noc(
                    device,
                    m,
                    n_cmds=n_cmds,
                    xfer_bytes=s,
                    barrier_each=False,
                    on_writer=False,
                    hgroups=h,
                    kgroups=k,
                    scratch=scratch,
                ),
                xs,
            )
            slope, intercept, r2 = cb.fit(xs, ys)
            pts = "  ".join(f"{s}B:{y:.0f}" for s, y in zip(xs, ys))
            gbps = (n_cmds / slope) if slope > 0 else float("inf")
            rows.append(
                f"  {h}x{k} {cb.NOC_MODES[mode]:<19} {slope*1000:7.2f} ps/B/core "
                f"({gbps:7.2f} GB/s/core)   size-0 intercept {intercept:8.0f} ns "
                f"=> {intercept/n_cmds:6.2f} ns/cmd   r2 {r2:.4f}   [{pts}]"
            )
    _report(f"NoC payload cost at fixed {n_cmds} commands (NCRISC, 1 barrier)", rows)


# ---------------------------------------------------------------------------
# PROBE 3 — semaphores
# ---------------------------------------------------------------------------
SEM_COUNTS = (32, 64, 128, 256, 512)


def test_sem_cost(device):
    rows = []
    for h, k in _grids(device):
        for on_writer in (False, True):
            risc = "BRISC/noc1" if on_writer else "NCRISC/noc0"
            for barrier_each in (True, False):
                tag = "bar/inc" if barrier_each else "1 barrier"
                xs = list(SEM_COUNTS)
                ys = _sweep(
                    device,
                    lambda n, b=barrier_each, w=on_writer: cb.run_sem_inc_p2p(
                        device, n_ops=n, barrier_each=b, on_writer=w, hgroups=h, kgroups=k
                    ),
                    xs,
                )
                rows.append(_fit_line(f"{h}x{k} inc_p2p {risc} {tag}", xs, ys, unit="inc"))
        xs = list(SEM_COUNTS)
        ys = _sweep(device, lambda n: cb.run_sem_wait_sat(device, n_ops=n, hgroups=h, kgroups=k), xs)
        rows.append(_fit_line(f"{h}x{k} wait_min (already satisfied)", xs, ys, unit="wait"))
    _report("semaphore unicast / wait cost", rows)


def test_sem_mcast_cost(device):
    rows = []
    for h, k in _grids(device):
        if h * k < 2:
            continue
        for barrier_each in (True, False):
            tag = "bar/inc" if barrier_each else "1 barrier"
            xs = list(SEM_COUNTS)
            ys = _sweep(
                device,
                lambda n, b=barrier_each: cb.run_sem_mcast(device, n_ops=n, barrier_each=b, hgroups=h, kgroups=k),
                xs,
            )
            rows.append(_fit_line(f"fan-out {h*k-1:>3} ({h}x{k}) {tag}", xs, ys, unit="mcast_inc"))
    _report("noc_semaphore_inc_multicast cost vs fan-out", rows)


def test_sem_incast_cost(device):
    """N cores incrementing ONE core's semaphore — the op's per-round ack pattern. Slope over ROUND
    count at fixed N is ns per round of an N-way incast; divide by N for ns per absorbed atomic."""
    gx, gy = cb.device_grid(device)
    max_cores = gx * gy
    rows = []
    xs = (16, 32, 64, 128)
    for n_senders in (8, 11, 44, 88, 110):
        if n_senders > max_cores:
            continue
        ys = _sweep(
            device,
            lambda n, s=n_senders: cb.run_sem_incast(device, n_rounds=n, n_senders=s, hgroups=gx, kgroups=gy),
            xs,
        )
        slope, intercept, r2 = cb.fit(xs, ys)
        pts = "  ".join(f"{n}:{y:.0f}" for n, y in zip(xs, ys))
        rows.append(
            f"  N={n_senders:>3} senders   {slope:8.2f} ns/round   {slope/n_senders:7.2f} ns/atomic   "
            f"intercept {intercept:7.0f} ns   r2 {r2:.4f}   [{pts}]"
        )
    _report("N-way semaphore INCAST (one target absorbing N atomics per round)", rows)


# ---------------------------------------------------------------------------
# PROBE 4 — the grid-wide multicast rendezvous
# ---------------------------------------------------------------------------
def test_mcast_rendezvous(device):
    """The op's phase-2 round with the payload removed. Slope over ROUNDS = ns per round.

    The peel: ACK_ONLY prices the ack incast with nothing serialising it; ACK_READY is the real,
    fully-ordered round minus the payload write; FULL adds the payload write back. FULL - ACK_READY
    is therefore the data multicast measured under IDENTICAL ordering. (ACK_DATA is left in as a
    diagnostic only — with no ready signal nothing orders the rounds, so it is a race and its fit is
    meaningless; it is reported, not read.)"""
    payload = 64
    stages = [cb.MCAST_STAGE_ACK, cb.MCAST_STAGE_ACK_READY, cb.MCAST_STAGE_FULL, cb.MCAST_STAGE_ACK_DATA]
    rows = []
    for h, k in _grids(device):
        if h * k < 4:
            continue
        for depth in (1, 2, 4):
            for stage in stages:
                xs = [2 * h, 4 * h, 8 * h, 16 * h]
                ys = _sweep(
                    device,
                    lambda r, s=stage, d=depth: cb.run_mcast(
                        device, rounds=r, stage=s, depth=d, payload_bytes=payload, hgroups=h, kgroups=k
                    ),
                    xs,
                )
                rows.append(_fit_line(f"{h}x{k} depth={depth} {cb.MCAST_STAGES[stage]:<19}", xs, ys, unit="round"))
    _report(f"grid-wide multicast rendezvous, {payload} B payload (slope = ns per round)", rows)


SAMPLES = 5


def test_mcast_focus(device):
    """FOCUS: the op's own configuration — 11x8, DEPTH_H=3, 11 rounds per M-block.

    SAMPLES runs per point, reported as median [min..max]. Two samples are not enough here: the
    DEPTH=1 (fully serialised) round is deterministic to 0.0%, but every DEPTH>1 row is genuinely
    variable — 88 receivers pipelining D rounds deep interleave differently run to run, and the
    spread reaches 30%. That variance is a property of the pipelined rendezvous, not of the
    measurement, so it gets reported as a RANGE rather than averaged into a single figure."""
    import statistics

    h, k = cb.clamp_grid(device, 11, 8)
    payload = 64
    rows = []
    for depth in (1, 2, 3, 4):
        for stage in (cb.MCAST_STAGE_ACK, cb.MCAST_STAGE_ACK_READY, cb.MCAST_STAGE_FULL):
            per_round = []
            for _ in range(SAMPLES):
                ns = cb.measure_once(
                    device,
                    lambda s=stage, d=depth: cb.run_mcast(
                        device, rounds=8 * h, stage=s, depth=d, payload_bytes=payload, hgroups=h, kgroups=k
                    ),
                )
                per_round.append(ns / (8 * h))
            med = statistics.median(per_round)
            spread = (max(per_round) - min(per_round)) / max(per_round) * 100
            rows.append(
                f"  depth={depth} {cb.MCAST_STAGES[stage]:<19} median {med:8.1f} "
                f"[{min(per_round):7.1f} .. {max(per_round):7.1f}] ns/round  (spread {spread:4.1f}%)   "
                f"=> {11 * med / 1000:6.2f} us per 11-round M-block"
            )
    _report(f"mcast rendezvous FOCUS: grid {h}x{k}, {8*h} rounds, {payload} B payload, {SAMPLES} samples", rows)


def test_mcast_payload_scaling(device):
    """Same rendezvous, sweeping the payload: separates the fixed rendezvous cost from the bytes."""
    h, k = cb.clamp_grid(device, 11, 8)
    rounds = 8 * h
    rows = []
    for depth in (1, 4):
        for payload in (32, 256, 1024, 4096, 16384):
            ns = cb.measure_once(
                device,
                lambda p=payload, d=depth: cb.run_mcast(
                    device, rounds=rounds, stage=cb.MCAST_STAGE_FULL, depth=d, payload_bytes=p, hgroups=h, kgroups=k
                ),
            )
            rows.append(f"  depth={depth} payload={payload:>6} B   {ns:9.0f} ns   {ns/rounds:8.1f} ns/round")
    _report(f"rendezvous vs payload, grid {h}x{k}, {rounds} rounds", rows)


# ---------------------------------------------------------------------------
# PROBE 5 — DEST sync
# ---------------------------------------------------------------------------
def test_dest_sync_cost(device):
    xs = (64, 128, 256, 512, 1024)
    rows = []
    for h, k in _grids(device):
        ys = _sweep(device, lambda n: cb.run_dest(device, n_iters=n, hgroups=h, kgroups=k), xs)
        rows.append(_fit_line(f"{h}x{k} tile_regs acquire/commit/wait/release", xs, ys, unit="cycle"))
    _report("DEST sync handshake, no math (slope = ns per tile_regs cycle)", rows)
