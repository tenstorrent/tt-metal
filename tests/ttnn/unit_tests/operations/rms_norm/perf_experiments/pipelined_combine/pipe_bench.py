# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off — idea I9: a cheaper, PIPELINED rendezvous for the rms_norm
cross-core statistics combine.

Reconstructs ONLY the collective: every core of a reduction rectangle starts with
`rows_t` bf16 stat tiles resident in L1 (a pinned HEIGHT shard — no reader, no DRAM, no
statistics pipeline, no apply pass) and the group must land rsqrt(sum + eps) in every
member's output shard. Forked from the round-1 `allgather_combine` bench so the
baseline is bit-for-bit the same reconstruction of the op's current gather.

Variants (see kernels/pipe_dataflow.cpp):
  baseline   the op's CURRENT rendezvous — per-contributor
             write + noc_async_write_barrier + semaphore.up, receiver does one
             all-or-nothing sem.wait_min(fan_in-1) then pushes the whole fan-in.
  flag       CHEAPER DELIVERY only: set_async_write_state + async_write_with_state for
             the data and a monotone inline_dw_write flag word instead of the NoC
             atomic. Still all-or-nothing on the receiving side.
  incr       flag delivery + PIPELINED ACCUMULATION: the receiver releases each
             contributor's slot as it lands and the combine chain consumes with
             WaitPolicy::Cumulative, overlapping the tile adds with the remaining
             arrivals. Both levels of the tree.

SKEW is a first-class axis. The op's contributors do not arrive together (round-1 zone
data: cp_wait_rstd min 735 / max 4371 ns), and a bench where all 110 cores arrive at
once cannot tell a pipelining idea from a null. Each core busy-waits a deterministic
per-core `skew_iters` before contributing; the pattern is IDENTICAL across variants.

  incr_sem   PIPELINED with the op's OWN atomic delivery: contributors still signal
             with one NoC atomic, but the increment is 1 << my_slot, so the single
             semaphore word carries per-contributor IDENTITY and the receiver can
             release slot j the moment bit j is set. Delivery cost identical to the
             baseline; only the release granularity changes.

Precision contract held fixed everywhere: bf16 stat pages, MathFidelity::HiFi2,
fp32_dest_acc_en=False. Nothing here trades precision for speed — the arithmetic is the
same chain over the same operands, only the arrival protocol moves. max_rel_err is
0.00138 for EVERY variant on every geometry (the bf16 rounding of the rsqrt output).

=============================================================================
MEASURED — blackhole_p150b (11x10 worker grid, 1.35 GHz), DEVICE KERNEL DURATION [ns],
one fresh dispatch per cell; the focus rows are the MEDIAN of 3 reps.
=============================================================================
geometry / skew           baseline   flag      incr      incr_sem
focus_11x10  skew=none      4104     3704 1.11x  3598 1.14x  4136 0.99x
focus_11x10  skew=mid       4550     4764 0.96x  4534 1.00x  4567 1.00x
focus_11x10  skew=big       6699     6801 0.98x  6590 1.02x  6544 1.02x
focus_11x10_b4 skew=none   14987    14667 1.02x 13829 1.08x 14925 1.00x
focus_11x10_b4 skew=big    24445    25282 0.97x 24604 0.99x 24431 1.00x
wshard_8x1   skew=none       2207     2267 0.97x  2148 1.03x  2176 1.01x
wshard_8x1   skew=big        5662     5738 0.99x  5616 1.01x  5661 1.00x
wshard_7x4   skew=none       2790     2930 0.95x  2854 0.98x  2743 1.02x
wshard_7x4   skew=big        5919     6070 0.98x  5919 1.00x  5870 1.01x
col_1x8      skew=none       2033     2099 0.97x  2036 1.00x  2005 1.01x
col_1x8      skew=big        5540     5567 1.00x  5434 1.02x  5409 1.02x
small_3x3    skew=none       2470     2601 0.95x  2492 0.99x  2464 1.00x
small_3x3    skew=big        5817     5916 0.98x  5917 0.98x  5854 0.99x
bshard_8x1_r16 skew=none    30594    30659 1.00x    —          —        (rows_t=16)
bshard_8x1_r16 skew=big     32267    32414 1.00x    —          —

READING OF THE RESULT.
The idea only pays in the ARTIFICIAL zero-skew regime (110 contributors issuing at the
same instant): there both the cheaper delivery and the pipelined release win ~1.1x on
the focus box, because what they relieve is ISSUE/POLL CONTENTION at a synchronized
start, not latency. Feed the bench the op's REAL arrival spread (round-1 zones:
cp_wait_rstd min 735 / max 4371 ns) and the win collapses to 1.00-1.02x, because a
gather's cost is set by the LAST contributor to arrive — a receive-side barrier that is
already satisfied by everyone else is free, and pipelining can only hide the ONE tile
add that still follows the last arrival (round 1 measured the whole combine arithmetic
at ~611 ns of the ~3050 ns combine region, and only the tail sliver of that is
hideable). The `flag` delivery is a small REGRESSION under skew: on Blackhole an inline
L1 dword write is bounced through an L1 scratch location (noc.h), so it costs the
critical (last) contributor more than the plain NoC atomic it replaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_ZERO = 4
CB_STAT_PARTIAL = 7
CB_STAT_GATHER = 8
CB_STAT_SUM = 9
CB_RSTD_SEND = 10
CB_STAT_GATHER2 = 15
CB_OUT = 16
CB_BRANCH_SUM = 18

SEM_GATHER = 0
SEM_MCAST_READY = 1
SEM_MCAST_CONSUMED = 2
SEM_GATHER2 = 3

MCAST_CT_BASE = 7
MCAST_RT_BASE = 15

MODE = {"baseline": 0, "flag": 5, "incr": 6, "incr_sem": 7}
VARIANTS = tuple(MODE)


@dataclass(frozen=True)
class Geom:
    nx: int
    ny: int
    rows_t: int = 1
    num_blocks: int = 1
    num_groups: int = 1

    @property
    def g(self):
        return self.nx * self.ny

    @property
    def tiles_per_core(self):
        return self.rows_t * self.num_blocks


GEOMS = {
    # (1,1,32,7168) INTERLEAVED — THE FOCUS SHAPE: G=110 in one 11x10 box, R=1, 1 block.
    "focus_11x10": Geom(11, 10, 1, 1, 1),
    "focus_11x10_b4": Geom(11, 10, 1, 4, 1),
    # WIDTH-sharded perf geometries.
    "wshard_8x1": Geom(8, 1, 1, 1, 1),
    "wshard_7x4": Geom(7, 4, 1, 1, 1),
    # BLOCK-sharded bshard1024: 8 concurrent 8-core groups, R=16 tile-rows, 2 blocks.
    "bshard_8x1_r16": Geom(8, 1, 16, 2, 8),
    "col_1x8": Geom(1, 8, 1, 1, 1),
    "small_3x3": Geom(3, 3, 1, 1, 1),
    "small_4x2": Geom(4, 2, 1, 1, 1),
    "small_4x2_r4_b2": Geom(4, 2, 4, 2, 1),
}

# Per-core arrival skew, expressed as the MAX spread in ns between the earliest and
# latest contributor. `none` is the "everyone arrives together" bench; `mid`/`big`
# bracket the op's measured spread (cp_wait_rstd min 735 / max 4371 ns => ~3.6 us).
SKEWS = {"none": 0, "mid": 1500, "big": 3600}

# Calibrated on blackhole_p150b @1.35 GHz: the busy loop below costs 5.91 ns/iteration
# (measured — 43600 iters => 257.3 us, 163500 iters => 967.9 us, both linear).
NS_PER_SKEW_ITER = 5.91


def _f32_bits(x):
    import struct

    return int(struct.unpack("<I", struct.pack("<f", float(x)))[0])


@dataclass
class Layout:
    geom: Geom
    cores: list
    core_ranges: ttnn.CoreRangeSet
    groups: list


def build_layout(device, geom: Geom) -> Layout:
    grid = device.compute_with_storage_grid_size()
    across = grid.x // geom.nx
    down = grid.y // geom.ny
    if across == 0 or down == 0 or geom.num_groups > across * down:
        raise ValueError(f"cannot place {geom.num_groups} groups of {geom.ny}x{geom.nx}")
    ranges, groups = [], []
    for gi in range(geom.num_groups):
        gx = (gi % across) * geom.nx
        gy = (gi // across) * geom.ny
        box = (gx, gy, gx + geom.nx - 1, gy + geom.ny - 1)
        cores = [(gx + x, gy + y) for y in range(geom.ny) for x in range(geom.nx)]
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(box[0], box[1]), ttnn.CoreCoord(box[2], box[3])))
        groups.append({"box": box, "cores": cores})
    core_ranges = ttnn.CoreRangeSet(ranges)
    cores = []
    for rng in core_ranges.ranges():
        for y in range(rng.start.y, rng.end.y + 1):
            for x in range(rng.start.x, rng.end.x + 1):
                cores.append((x, y))
    return Layout(geom, cores, core_ranges, groups)


def make_tensors(device, geom: Geom, layout: Layout, torch_partials):
    import torch

    n = len(layout.cores)
    t = geom.tiles_per_core
    in_torch = torch.zeros((n * 32 * t, 32), dtype=torch.float32)
    for c in range(n):
        for j in range(t):
            in_torch[(c * t + j) * 32 : (c * t + j + 1) * 32, :] = float(torch_partials[c][j])

    def sharded(rows_per_core):
        return ttnn.create_sharded_memory_config(
            shape=(rows_per_core, 32),
            core_grid=layout.core_ranges,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    tt_in = ttnn.from_torch(
        in_torch.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded(32 * t),
    )
    # Page 0 = the combine's identity operand, page 1 = the arrival flag words. HOST
    # zeroed, which is what makes the monotone flag protocol race-free (a fast sender
    # may set a flag before the receiving core's kernel starts).
    tt_zero = ttnn.from_torch(
        torch.zeros((n * 32 * 2, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded(64),
    )
    tt_out = ttnn.from_torch(
        torch.zeros((n * 32 * t, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded(32 * t),
    )
    return tt_in, tt_zero, tt_out


def _cb(index, core_ranges, num_pages, page_size, data_format):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def _pinned_cb(index, tensor, core_ranges):
    desc = ttnn.cb_descriptor_from_sharded_tensor(index, tensor)
    desc.core_ranges = core_ranges
    return desc


def skew_iters_for(idx, geom, skew_ns):
    """Deterministic per-core arrival skew — a fixed pseudo-random spread of `skew_ns`."""
    if not skew_ns:
        return 0
    span = max(geom.g - 1, 1)
    frac = ((idx * 7 + 3) % (span + 1)) / span
    return int(round(skew_ns * frac / NS_PER_SKEW_ITER))


def create_program_descriptor(device, geom, layout, tt_in, tt_zero, tt_out, variant, skew=0, epsilon=1e-6):
    mode = MODE[variant]
    s1, s2 = (geom.nx, geom.ny) if (geom.nx > 1 and geom.ny > 1) else (geom.g, 1)
    if mode in (6, 7) and geom.rows_t > 1:
        # The gather CB is laid out (r * fan_in + slot); a per-slot incremental push is
        # only front-contiguous when rows_t == 1. Reported as a domain restriction, not
        # papered over.
        raise ValueError("incremental variants require rows_t == 1 (gather CB layout is row-major over slots)")
    g = geom.g
    rows_t, nb = geom.rows_t, geom.num_blocks
    tile_bytes = tt_in.buffer_aligned_page_size()
    fmt = tt_in.dtype

    mcast_cfg = ttnn.McastConfig(noc=ttnn.NOC.NOC_1, sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED])
    mcasts = []
    for grp in layout.groups:
        x0, y0, x1, y1 = grp["box"]
        rect = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])
        root = ttnn.CoreCoord(x1, y1)
        mcasts.append(ttnn.Mcast2D(device, rect, root, mcast_cfg, g - 1))
    mcast_ct = list(mcasts[0].compile_time_args(None))

    df_ct = [mode, s1, s2, g, SEM_GATHER, SEM_GATHER2, 0]
    assert len(df_ct) == MCAST_CT_BASE
    df_ct += mcast_ct
    cp_ct = [mode, s1, s2, g, _f32_bits(epsilon)]

    df_rt = ttnn.RuntimeArgs()
    cp_rt = ttnn.RuntimeArgs()
    for gi, grp in enumerate(layout.groups):
        x0, y0, x1, y1 = grp["box"]

        def virt(x, y):
            c = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            return int(c.x), int(c.y)

        vx0, vy0 = virt(x0, y0)
        vx1, vy1 = virt(x1, y1)
        root_vx, root_vy = virt(x1, y1)
        for idx, (cx, cy) in enumerate(grp["cores"]):
            row = idx // geom.nx
            col = idx % geom.nx
            is_root = 1 if (cx, cy) == (x1, y1) else 0
            if s2 > 1:
                leader_x, leader_y = virt(x1, cy)
                is_leader = 1 if cx == x1 else 0
                slot = col
            else:
                leader_x, leader_y = root_vx, root_vy
                is_leader = is_root
                slot = idx
            core_rt = [
                rows_t,
                nb,
                slot,
                is_leader,
                is_root,
                row,
                skew_iters_for(idx, geom, skew),
                leader_x,
                leader_y,
                root_vx,
                root_vy,
                vx0,
                vy0,
                vx1,
                vy1,
            ] + list(mcasts[gi].runtime_args(ttnn.CoreCoord(cx, cy)))
            assert len(core_rt) == MCAST_RT_BASE + 4, f"runtime-arg layout drifted: {len(core_rt)}"
            df_rt[cx][cy] = core_rt
            cp_rt[cx][cy] = [rows_t, nb, is_leader, is_root]

    crs = layout.core_ranges
    cbs = [
        _pinned_cb(CB_STAT_PARTIAL, tt_in, crs),
        _pinned_cb(CB_ZERO, tt_zero, crs),
        _pinned_cb(CB_OUT, tt_out, crs),
        _cb(CB_STAT_GATHER, crs, rows_t * s1, tile_bytes, fmt),
        _cb(CB_STAT_SUM, crs, rows_t, tile_bytes, fmt),
        _cb(CB_RSTD_SEND, crs, rows_t, tile_bytes, fmt),
    ]
    if s2 > 1:
        cbs.append(_cb(CB_STAT_GATHER2, crs, rows_t * s2, tile_bytes, fmt))
        cbs.append(_cb(CB_BRANCH_SUM, crs, rows_t, tile_bytes, fmt))
    semaphores = [
        ttnn.SemaphoreDescriptor(id=i, core_ranges=crs, initial_value=0)
        for i in (SEM_GATHER, SEM_MCAST_READY, SEM_MCAST_CONSUMED, SEM_GATHER2)
    ]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "pipe_dataflow.cpp"),
            core_ranges=crs,
            compile_time_args=df_ct,
            runtime_args=df_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "pipe_compute.cpp"),
            core_ranges=crs,
            compile_time_args=cp_ct,
            runtime_args=cp_rt,
            # The user's precision contract, identical in every variant.
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


def partials_for(geom, layout, kind="exact"):
    n = len(layout.cores)
    t = geom.tiles_per_core
    if kind == "exact":
        return [[float(2 ** (j % 3)) for j in range(t)] for _ in range(n)]
    return [[float(2 ** (j % 3)) * (1.0 + 0.25 * ((c % 4) + 1)) for j in range(t)] for c in range(n)]


def reference(geom, layout, partials, epsilon=1e-6):
    per_group = {}
    for gi, grp in enumerate(layout.groups):
        idx = [layout.cores.index(c) for c in grp["cores"]]
        per_group[gi] = [sum(partials[i][j] for i in idx) for j in range(geom.tiles_per_core)]
    out = [None] * len(layout.cores)
    for gi, grp in enumerate(layout.groups):
        for c in grp["cores"]:
            out[layout.cores.index(c)] = [(s + epsilon) ** -0.5 for s in per_group[gi]]
    return out


def run(device, geom_name, variant, partial_kind="exact", skew=0, epsilon=1e-6):
    """One dispatch. Returns (device_kernel_ns, max_rel_err)."""
    import torch

    geom = GEOMS[geom_name]
    layout = build_layout(device, geom)
    partials = partials_for(geom, layout, partial_kind)
    tt_in, tt_zero, tt_out = make_tensors(device, geom, layout, partials)
    desc = create_program_descriptor(device, geom, layout, tt_in, tt_zero, tt_out, variant, skew, epsilon)

    ttnn.ReadDeviceProfiler(device)
    out = ttnn.generic_op([tt_in, tt_zero, tt_out], desc)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    ns = None
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get("DEVICE KERNEL DURATION [ns]")
            if entry is not None:
                d = float(entry.duration)
                ns = d if ns is None else max(ns, d)

    got = ttnn.to_torch(out).to(torch.float32)
    exp = reference(geom, layout, partials, epsilon)
    t = geom.tiles_per_core
    max_rel = 0.0
    for c in range(len(layout.cores)):
        for j in range(t):
            # rstd is consumed as a Col broadcast: column 0 alone is finalized by design.
            v = float(got[(c * t + j) * 32, 0])
            e = exp[c][j]
            max_rel = max(max_rel, abs(v - e) / abs(e))
    return ns, max_rel
