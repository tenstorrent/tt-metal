# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: rms_norm's CROSS-CORE COMBINE, flat root vs reduce-scatter.

WHAT IS ISOLATED
    One row-group of `s` cores splits a hidden axis of `s*S` tiles.  Each core holds
    its own resident L1 shard of `nb*B` tile-rows x `S` tiles and must end up with
    the finalized `1/rms` of every row of every block.  That is the whole bench:

        Sum(x*x) per core  ->  cross-core combine  ->  1/rms resident on every core

    Held trivial on purpose (they are identical in every variant and are NOT the
    concept under test): no gamma, no mask, no tilize, no scale/apply pass, and no
    DRAM at all — x is a resident L1 shard and the broadcast lands DIRECTLY in the
    output shard, so "the stat reached this core" costs zero extra copies.

VARIANTS (`flat_root` is the shipped op's approach — the honest baseline)
    flat_root      one root gathers s*B stat tiles, reduces ALL B rows, mcasts B tiles.
    scatter_root   min(s,B) owners each reduce their OWN rows, funnel the finalized
                   rows to the root, root mcasts B tiles (SAME mcast_pipe broadcast
                   as the baseline -> isolates "scatter the reduce work").
    scatter_mcast  same scatter, each owner broadcasts its own rows directly to the
                   group; every core waits for a COUNT of arrivals (raw mcast +
                   counted semaphore, because ReceiverPipe cannot wait a count).
    scatter_mcast_barrier
                   scatter_mcast + a per-block group barrier, which PRICES the
                   landing-buffer reuse protocol the real op would need (its
                   cb_rms_recip is B pages and is popped every block).

PRECISION CONTRACT — FIXED, not a lever: bf16 activations, float32 stat tiles,
math_fidelity=HiFi2, fp32_dest_acc_en=False, math_approx_mode=False.  Every variant
runs under the identical config; the stat format is never shrunk.  Every variant also
lands on the SAME numbers to the last digit (sum-then-collapse == collapse-then-sum,
and the scatter only moves WHICH core adds the same s partials): per case, all
variants report an identical worst relative error vs an fp32 torch reference.

MEASURED — Blackhole p150b @1350 MHz, device kernel ns, one fresh run per point
(the focus / s8_B1 / s28_B8 rows were repeated: spread < 0.5%).  S=4 throughout.
`usig` = the gather signal is one unicast atomic per owner; the plain `scatter_root`
column uses one MULTICAST atomic instead, which is the trap described below.

    case      geometry                     flat_root  scatter_root  scatter_root_usig
    s2_B8     W=256   32 groups, 4 owners      46190       36239      36107   1.28x
    s4_B8     W=512   16 groups, 2 owners      56924       30959      30470   1.87x
    FOCUS     W=1024   8 groups, 8 owners      64965       30313      28890   2.25x
    s16_B8    W=2048   4 groups, 8 owners      81386       99127      40524   2.01x
    s28_B8    W=3584   2 groups, 8 owners      53129      127061      26260   2.02x
    s32_B8    W=4096   2 groups, 8 owners      57105      149716      31073   1.84x
    s8_B4     W=1024   8 groups, 4 owners      35090       30636      20787   1.69x
    s8_B32    W=1024   8 groups, 8 owners  INFEASIBLE      24919      24956      —
    s8_B1     W=1024   8 groups, 1 owner       12545       13707      13676   0.92x
    s32_B1    W=4096   2 groups, 1 owner       19589       20607      20673   0.95x

Three things the table says:

 1. Scattering the reduce is worth 1.3-2.25x whenever a block has more than one
    tile-row.  At B == 1 there is nothing to scatter (num_owners == 1) and the
    scatter degenerates to the flat root PLUS one funnel hop — a measured -8%.  So the
    op should keep the flat root exactly when `min(s, B) == 1`.

 2. `noc_semaphore_inc_multicast` is a TRAP for the gather signal.  One multicast
    atomic per contributor looks O(1) in s against num_owners unicast atomics, and it
    is 1.1x faster at s=8 in isolation — but its multicast PATH RESERVATION serializes
    against every other group's, and on a 2-D row-group it turns the win into a
    0.38-0.82x loss (s=16: 99127 vs 40524; s=32: 149716 vs 31073).  Same topology,
    same bytes, same correctness — only the signal differs.

 3. The flat root's gather CB is `s*B` float32 stat pages; the scatter's is
    `s*(B/min(s,B))`.  At B=32 that is 1 MB vs 32 KB, and the baseline does not FIT
    L1 (s8_B32) — so the scatter also unlocks coarser blocks, which the op's own
    sharded measurements wanted (block_rows 16 beat 8 but was unaffordable).

And what the s-sender broadcast said, on the same sweep (both with the unicast gather
signal, so the only difference is the broadcast): each owner MULTICASTING its own rows
to the group is 1.04-1.22x (focus 57219 ns, 1.14x) — s concurrent one-tile multicasts
over the same line each pay a path reservation and never amortize it, the same
mechanism as (2).  Each owner UNICASTING its rows to the group is 1.15-1.81x (focus
35902 ns) — better, but still below funnel-to-root everywhere.  A per-block group
barrier (the s-sender
stand-in for mcast_pipe's consumer-ready ack) costs another 1.4x on top
(scatter_mcast_barrier 73352, scatter_ucast_barrier 51753).  Hence the recommendation
keeps the SINGLE-sender broadcast the op already ships, whose ack is free (flat_root
64980 vs flat_root_hs 65057; scatter_root 33784 vs scatter_root_hs 32184).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import struct
import torch

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
EPS = 1e-6

# CB slots (semantic names; the numeric slot is only the buffer index)
CB_IN = 0
CB_SQ_PARTIALS = 2
CB_GATHERED = 4
CB_STAT_OUT = 5
CB_RMS_RECIP = 6
CB_SCALER = 7
CB_BCAST_STAGE = 8

SEM_MCAST_READY = 0
SEM_MCAST_CONSUMED = 1
SEM_GATHER = 2
SEM_BCAST = 3
SEM_STAT_READY = 4
SEM_DRAIN = 5

V_FLAT_ROOT = 0
V_SCATTER_ROOT = 1
V_SCATTER_MCAST = 2
V_SCATTER_UCAST = 3

# (topology, price the reuse protocol, mcast_pipe PRE_HANDSHAKE, unicast gather signal)
#
# The `_hs` pair is the reuse protocol as the SHIPPED op spells it: mcast_pipe's
# receiver->sender readiness ack, which the real op turns on whenever a row-group has
# more than one block (its cb_rms_recip is B pages and is popped every block).  The
# `_barrier` pair is the s-sender equivalent (a per-block group barrier), since no
# single sender can be acked when there are s of them.
VARIANTS = {
    "flat_root": (V_FLAT_ROOT, False, False, False),
    "flat_root_hs": (V_FLAT_ROOT, False, True, False),
    "scatter_root": (V_SCATTER_ROOT, False, False, False),
    "scatter_root_hs": (V_SCATTER_ROOT, False, True, False),
    "scatter_root_usig": (V_SCATTER_ROOT, False, False, True),
    "scatter_mcast": (V_SCATTER_MCAST, False, False, False),
    "scatter_mcast_usig": (V_SCATTER_MCAST, False, False, True),
    "scatter_ucast": (V_SCATTER_UCAST, False, False, False),
    "scatter_ucast_usig": (V_SCATTER_UCAST, False, False, True),
    "scatter_mcast_barrier": (V_SCATTER_MCAST, True, False, False),
    "scatter_ucast_barrier": (V_SCATTER_UCAST, True, False, False),
}


def _f32_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", float(value)))[0]


@dataclass(frozen=True)
class Geo:
    """s slices x S hidden tiles, blocks of B rows, nb blocks, on a gw x gh grid."""

    s: int
    S: int
    B: int
    nb: int
    gw: int
    gh: int

    @property
    def shard_rows(self) -> int:
        return self.nb * self.B

    @property
    def width(self) -> int:
        return self.s * self.S * TILE

    @property
    def label(self) -> str:
        return f"s{self.s}_S{self.S}_B{self.B}_nb{self.nb}"


@dataclass
class Plan:
    geo: Geo
    grid: "ttnn.CoreRangeSet"
    cores: list  # logical (x, y) in shard order
    groups: list  # list of dicts: cores, bbox_logical, bbox_virtual, root_virtual


def plan(device, geo: Geo) -> Plan:
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(geo.gw - 1, geo.gh - 1))])
    cores = [(int(c.x), int(c.y)) for c in ttnn.corerange_to_cores(grid, None, True)]
    assert len(cores) % geo.s == 0, f"{len(cores)} cores is not a multiple of s={geo.s}"
    groups = []
    for r in range(len(cores) // geo.s):
        gcores = cores[r * geo.s : (r + 1) * geo.s]
        xs = [c[0] for c in gcores]
        ys = [c[1] for c in gcores]
        span = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
        assert span == geo.s, (
            f"row-group {r} {gcores} is not a rectangle (bbox holds {span} cores, s={geo.s}); "
            "pick a (gw, gh) where s tiles the row-major core order"
        )
        virt = [device.worker_core_from_logical_core(ttnn.CoreCoord(x, y)) for x, y in gcores]
        vx = [int(v.x) for v in virt]
        vy = [int(v.y) for v in virt]
        groups.append(
            {
                "cores": gcores,
                "bbox_logical": (min(xs), min(ys), max(xs), max(ys)),
                "bbox_virtual": (min(vx), min(vy), max(vx), max(vy)),
                "root_virtual": (int(virt[0].x), int(virt[0].y)),
                "virtual": list(zip(vx, vy)),
            }
        )
    return Plan(geo=geo, grid=grid, cores=cores, groups=groups)


def make_tensors(device, p: Plan, *, seed: int = 42):
    """Resident L1 input shards + the (allocated) output stat shards + torch truth."""
    geo = p.geo
    rows = geo.shard_rows * TILE
    ngroups = len(p.groups)
    torch.manual_seed(seed)
    x_groups = torch.randn(ngroups, rows, geo.width, dtype=torch.float32).to(torch.bfloat16)

    slice_w = geo.S * TILE
    bands = []
    for idx in range(len(p.cores)):
        r, c = divmod(idx, geo.s)
        bands.append(x_groups[r][:, c * slice_w : (c + 1) * slice_w])
    flat = torch.cat(bands, dim=0).reshape(1, 1, len(p.cores) * rows, slice_w)

    in_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(p.grid, [rows, slice_w], ttnn.ShardOrientation.ROW_MAJOR),
    )
    x = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mc)

    out_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(p.grid, [rows, TILE], ttnn.ShardOrientation.ROW_MAJOR),
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, len(p.cores) * rows, TILE]), ttnn.float32, ttnn.TILE_LAYOUT, device, out_mc
    )

    xf = x_groups.to(torch.float32)
    expected = torch.rsqrt(xf.pow(2).mean(dim=-1) + EPS)  # (ngroups, rows)
    return x, out, expected


def check(p: Plan, out, expected):
    """Every core must hold the finalized 1/rms of EVERY row of its row-group."""
    geo = p.geo
    rows = geo.shard_rows * TILE
    got = ttnn.to_torch(out).to(torch.float32).reshape(-1, TILE)
    worst_rel = 0.0
    acc_a, acc_b = [], []
    for idx in range(len(p.cores)):
        r = idx // geo.s
        col0 = got[idx * rows : (idx + 1) * rows, 0]
        ref = expected[r]
        worst_rel = max(worst_rel, ((col0 - ref).abs() / ref.abs()).max().item())
        acc_a.append(col0)
        acc_b.append(ref)
    a = torch.cat(acc_a)
    b = torch.cat(acc_b)
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return pcc, worst_rel


def build_program(device, p: Plan, x, out, *, variant: str, compute_config):
    geo = p.geo
    vcode, price_barrier, handshake, signal_unicast = VARIANTS[variant]

    num_owners = 1 if vcode == V_FLAT_ROOT else min(geo.s, geo.B)
    assert geo.B % num_owners == 0, f"B={geo.B} must divide by num_owners={num_owners}"
    own_rows = geo.B // num_owners

    in_tile = ttnn.tile_size(ttnn.bfloat16)
    stat_tile = ttnn.tile_size(ttnn.float32)
    bf16_tile = ttnn.tile_size(ttnn.bfloat16)

    def _cb(index, pages, page_size, dtype):
        return ttnn.CBDescriptor(
            total_size=pages * page_size,
            core_ranges=p.grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, x),
        ttnn.cb_descriptor_from_sharded_tensor(CB_RMS_RECIP, out),
        _cb(CB_SQ_PARTIALS, geo.B, stat_tile, ttnn.float32),
        _cb(CB_GATHERED, geo.s * own_rows, stat_tile, ttnn.float32),
        _cb(CB_STAT_OUT, own_rows, stat_tile, ttnn.float32),
        _cb(CB_SCALER, 1, bf16_tile, ttnn.bfloat16),
    ]
    if vcode == V_SCATTER_ROOT:
        # Landing buffer for the owners' finalized rows on the root.  Sized to the
        # WHOLE run (nb*B pages) so no block reuses another's pages — the bench
        # measures topology, not a reuse protocol.
        cbs.append(_cb(CB_BCAST_STAGE, geo.nb * geo.B, stat_tile, ttnn.float32))

    # ---- mcast wire (baseline + scatter_root only; scatter_mcast is raw) ----
    mcast_by_group = {}
    if vcode in (V_FLAT_ROOT, V_SCATTER_ROOT):
        cfg = ttnn.McastConfig(
            noc=ttnn.NOC.NOC_0,
            # The bench's landing buffer is per-block-disjoint, so the ack is not
            # needed for correctness here — it is a PRICED option (`*_hs`), because the
            # real op's B-page cb_rms_recip does need it.
            handshake=handshake,
            sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED],
        )
        for gi, g in enumerate(p.groups):
            ox, oy = g["cores"][0]
            rect_crs = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(g["bbox_logical"][0], g["bbox_logical"][1]),
                        ttnn.CoreCoord(g["bbox_logical"][2], g["bbox_logical"][3]),
                    )
                ]
            )
            mcast_by_group[gi] = ttnn.Mcast2D(device, rect_crs, ttnn.CoreCoord(ox, oy), cfg, geo.s - 1)
        mcast_ct = list(mcast_by_group[0].compile_time_args())
    else:
        mcast_ct = [0, 0, 0, 0, 0]
    assert len(mcast_ct) == 5

    reader_ct = list(mcast_ct) + [
        geo.S,
        geo.B,
        geo.s,
        own_rows,
        num_owners,
        vcode,
        stat_tile,
        SEM_GATHER,
        SEM_BCAST,
        SEM_STAT_READY,
        SEM_DRAIN,
        geo.shard_rows * geo.S,  # IN_WAIT_TILES — the whole resident shard
        1 if price_barrier else 0,
    ]
    writer_ct = [
        geo.B,
        geo.s,
        own_rows,
        num_owners,
        vcode,
        stat_tile,
        SEM_GATHER,
        1 if signal_unicast else 0,
    ]
    compute_ct = [geo.S, geo.B, geo.s, own_rows, geo.shard_rows * geo.S]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    inv_w_bits = _f32_bits(1.0 / float(geo.width))
    eps_bits = _f32_bits(EPS)

    for gi, g in enumerate(p.groups):
        vxlo, vylo, vxhi, vyhi = g["bbox_virtual"]
        root_x, root_y = g["root_virtual"]
        mc = mcast_by_group.get(gi)
        owner_coords = []
        for o in range(num_owners):
            owner_coords.extend(g["virtual"][o])
        peer_coords = []
        for c in range(geo.s):
            peer_coords.extend(g["virtual"][c])
        for slice_index, (cx, cy) in enumerate(g["cores"]):
            is_root = 1 if slice_index == 0 else 0
            is_owner = 1 if slice_index < num_owners else 0
            mcast_rt = list(mc.runtime_args(ttnn.CoreCoord(cx, cy))) if mc is not None else [0, 0, 0, 0]
            reader_rt[cx][cy] = (
                mcast_rt
                + [
                    geo.nb,
                    is_root,
                    is_owner,
                    slice_index * own_rows,
                    vxlo,
                    vylo,
                    vxhi,
                    vyhi,
                    geo.s - 1,
                    geo.s,
                    root_x,
                    root_y,
                ]
                + peer_coords
            )
            writer_rt[cx][cy] = [
                geo.nb,
                slice_index,
                root_x,
                root_y,
                vxlo,
                vylo,
                vxhi,
                vyhi,
                geo.s - 1,
            ] + owner_coords
            compute_rt[cx][cy] = [geo.nb, is_owner, inv_w_bits, eps_bits]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "combine_reader.cpp"),
            core_ranges=p.grid,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "combine_writer.cpp"),
            core_ranges=p.grid,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "combine_compute.cpp"),
            core_ranges=p.grid,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=compute_config,
        ),
    ]
    semaphores = [
        ttnn.SemaphoreDescriptor(id=i, core_ranges=p.grid, initial_value=0)
        for i in (SEM_MCAST_READY, SEM_MCAST_CONSUMED, SEM_GATHER, SEM_BCAST, SEM_STAT_READY, SEM_DRAIN)
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


def target_compute_config():
    """The perf group's FIXED precision contract — identical for every variant."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=False,
        math_approx_mode=False,
    )


def run_variant(device, p: Plan, x, out, *, variant: str):
    pd = build_program(device, p, x, out, variant=variant, compute_config=target_compute_config())
    return ttnn.generic_op([x, out], pd)
