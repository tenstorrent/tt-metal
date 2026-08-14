# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: COALESCE THE GATHER WRITES of rms_norm's cross-core combine.

WHAT IS ISOLATED
    One row-group of `s` cores splits a hidden axis of `s*S` tiles.  Each core holds
    its own resident L1 shard of `nb*B` tile-rows x `S` tiles and must end up with
    the finalized `1/rms` of every row of every block.  That is the whole bench:

        Sum(x*x) per core  ->  reduce-scatter combine  ->  1/rms resident on every core

    Held trivial on purpose (identical in every variant, NOT the concept under test):
    no gamma, no mask, no tilize, no scale/apply pass, and no DRAM at all — x is a
    resident L1 shard and the broadcast lands DIRECTLY in the output shard.  The
    TOPOLOGY is also held constant: every variant runs the shipped op's Perf-1
    reduce-scatter (num_owners = min(s, B) owners, unicast gather atomics, owners
    funnel their finalized rows to the root, root multicasts via mcast_pipe).

THE ONE KNOB
    The landing-page map of the gather, which sets the TRANSACTION COUNT for the
    same bytes:

      row-major (shipped)  page(r) = (r % own_rows) * s + slice_index
                           -> a contributor's own_rows tiles for ONE owner are
                              `s` pages apart: B writes of STAT_TILE_BYTES.
      coalesced (candidate) page(r) = slice_index * own_rows + (r % own_rows)
                           -> those tiles are contiguous at BOTH ends:
                              num_owners writes of own_rows * STAT_TILE_BYTES.

    Same bytes, same destinations, same arithmetic.  At own_rows == 1 the two maps
    are ALGEBRAICALLY IDENTICAL (both reduce to page = slice_index), so that regime
    is an exact no-op, not an approximation.

THE COST THE CANDIDATE PAYS
    The coalesced landing buffer is (slice outer, own_row inner) — the transpose of
    what `compute_kernel_lib::reduce` can index.  See the RAW-LLK JUSTIFICATION block
    at the top of kernels/gc_compute.cpp.  `baseline_raw` is the control that prices
    that bypass on the SHIPPED page map, so `coalesce - baseline_raw` is the NoC
    change alone and `baseline_raw - baseline` is the helper-bypass cost alone.

PRECISION CONTRACT — FIXED, not a lever: bf16 activations, float32 stat tiles,
math_fidelity=HiFi2, fp32_dest_acc_en=False, math_approx_mode=False.  Every variant
runs under the identical ComputeConfigDescriptor.  The candidate's arithmetic is
bit-identical to the baseline by construction: the same s partials are summed for the
same output row in the same pairwise order, only their tile INDICES are relabeled.
The test MEASURES that (`drift_vs_baseline`) rather than assuming it: 0.0 everywhere
except the s=2 cases, where the raw combine forces AccumulateViaAdd over the helper's
ReduceTile (see below) — a datapath change, not a paging change.

MEASURED — Blackhole p150b @1350 MHz, device kernel ns, one fresh run per point.
S=4 and shard_rows = nb*B = 32 throughout, so every case is the same total work.

    case      geometry              txns/blk  baseline  baseline_raw  coalesce
    focus     s8  B16 own_rows=2    16 -> 8      26156         26067     26150  1.000x
    s2_B16    s2  B16 own_rows=8    16 -> 2      33881         36517     37064  0.914x
    s4_B16    s4  B16 own_rows=4    16 -> 4      28802         28610     28249  1.020x
    s16_B16   s16 B16 own_rows=1    16 -> 16     32416         32161     31701  1.023x
    s28_B16   s28 B16 own_rows=1    16 -> 16     37122         37006     36942  1.005x
    s32_B16   s32 B16 own_rows=1    16 -> 16     41588         41366     41640  0.999x
    s8_B1     s8  B1  own_rows=1     1 -> 1     102306        101483    101358  1.009x
    s8_B8     s8  B8  own_rows=1     8 -> 8      29025         28891     28902  1.004x
    s8_B32    s8  B32 own_rows=4    32 -> 8      25019         24714     24819  1.008x
    s4_B32    s4  B32 own_rows=8    32 -> 4      27693         27525     27698  1.000x

NULL everywhere.  The ONE outlier, s2_B16 at 0.914x, is NOT the paging: baseline_raw
(same paging, raw combine) is already at 0.928x.  At s < 4 the op picks
ReduceAlgorithm::Auto (ReduceTile / matmul-with-ones) and the raw strided combine is
necessarily AccumulateViaAdd, which is the slower datapath there — which is exactly
why the op has that s >= 4 crossover.  coalesce/baseline_raw at s2 is 0.985x: flat.

WHY IT IS NULL — the stage is BYTE-bound, not transaction-bound.  Sweeping the
transaction count 8x at fixed bytes and fixed destinations (`*_split2/4` cut every
write into pieces; `coalesce` merges them) moves the writer barely at all
(ns/core over both blocks, focus):

    txns/blk/core   bytes/txn   wr_gather_issue   wr_gather_barrier   issue+barrier
        8 (coalesce)    8192           6207                868             7075
       16 (baseline)    4096           6577                531             7108
       64 (split4)      1024           7632                238             7870

    -> ~13 ns per transaction of RISC/command cost, on top of a ~3000 ns-per-block
       floor that does not depend on the transaction count at all.  64 KB of stat
       tiles per core per block at ~22 GB/s per core (~1.4 TB/s aggregate over 64
       cores) IS that floor.  Coalescing buys 33 ns of a 7108 ns stage, and the
       barrier gives most of it back draining a longer tail transaction.

    The "206 ns per 4 KB write" the whole-op zone report shows is 3289/16 — a
    division artifact.  Only ~13 ns of it is per-transaction; the other ~193 is the
    bytes, which coalescing does not touch.

Also measured at focus, same conclusion from the other directions:
    baseline_chunk4   26459 (0.988x)   coalesce_chunk2 26354 (0.992x)
    coalesce_chunk4   26153 (1.000x)   -- chunked barriers on the burst: null/negative
    flat_baseline     64916            flat_baseline_raw 64373
    flat_coalesce     64216 (1.011x)   -- num_owners == 1, B=16: 16 x 4 KB -> one
                                          64 KB write (which noc_async_write itself
                                          splits into 4 x 16 KB, NOC_MAX_BURST_SIZE).
                                          Still flat, and the flat root is 2.5x slower
                                          than the shipped reduce-scatter anyway.

FIDELITY OF THE ISOLATION: this bench's `wr_gather_issue` is 6577 ns/core with
max/p50 = 1.45; the whole op's is 6604 ns/core with max/p50 = 1.44.  The stage is
reproduced, so the null is about the stage and not about the harness.

WHERE THE HEADROOM ACTUALLY IS: the gather is byte-bound, so the only lever on it is
fewer BYTES (a compacted partial), not fewer transactions.
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
SEM_STAT_READY = 4

PAGING_ROWMAJOR = 0
PAGING_COALESCED = 1

COMBINE_HELPER = 0
COMBINE_RAW = 1

# name -> (paging, combine_impl, gather_chunk, force_flat, split)
#
#   baseline        the SHIPPED op: row-major landing, ckl::reduce helper.
#   baseline_raw    control — same landing map, combine hand-rolled as the helper's
#                   own AccumulateViaAdd walk (stride 1).  Prices the helper bypass.
#   coalesce        the idea: coalesced landing map + the same hand-rolled combine
#                   with (row_pitch, reduce_stride) = (1, own_rows).
#   *_chunkN        a noc_async_write_barrier every N gather transactions (the
#                   writer-side analogue of the op's DM_CHUNK_TILES store batching).
#   flat_*          num_owners forced to 1 (the pre-Perf-1 flat-root decode): the
#                   coalesced form there is ONE B-tile-wide write per contributor.
#   *_split2/4      the OPPOSITE lever — every transaction cut into 2/4 pieces, same
#                   bytes and destinations.  The control that says whether this stage
#                   is priced per TRANSACTION or per BYTE.
VARIANTS = {
    "baseline": (PAGING_ROWMAJOR, COMBINE_HELPER, 0, False, 1),
    "baseline_raw": (PAGING_ROWMAJOR, COMBINE_RAW, 0, False, 1),
    "coalesce": (PAGING_COALESCED, COMBINE_RAW, 0, False, 1),
    "baseline_chunk4": (PAGING_ROWMAJOR, COMBINE_HELPER, 4, False, 1),
    "coalesce_chunk2": (PAGING_COALESCED, COMBINE_RAW, 2, False, 1),
    "coalesce_chunk4": (PAGING_COALESCED, COMBINE_RAW, 4, False, 1),
    "flat_baseline": (PAGING_ROWMAJOR, COMBINE_HELPER, 0, True, 1),
    "flat_baseline_raw": (PAGING_ROWMAJOR, COMBINE_RAW, 0, True, 1),
    "flat_coalesce": (PAGING_COALESCED, COMBINE_RAW, 0, True, 1),
    "baseline_split2": (PAGING_ROWMAJOR, COMBINE_HELPER, 0, False, 2),
    "baseline_split4": (PAGING_ROWMAJOR, COMBINE_HELPER, 0, False, 4),
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
    cores: list
    groups: list


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
        assert span == geo.s, f"row-group {r} {gcores} is not a rectangle (bbox {span}, s={geo.s})"
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
    expected = torch.rsqrt(xf.pow(2).mean(dim=-1) + EPS)
    return x, out, expected


def read_stats(p: Plan, out):
    """Every core's view of its row-group's finalized 1/rms, as a flat fp32 vector."""
    geo = p.geo
    rows = geo.shard_rows * TILE
    got = ttnn.to_torch(out).to(torch.float32).reshape(-1, TILE)
    return torch.stack([got[i * rows : (i + 1) * rows, 0] for i in range(len(p.cores))])


def check(p: Plan, out, expected):
    """Every core must hold the finalized 1/rms of EVERY row of its row-group."""
    geo = p.geo
    got = read_stats(p, out)
    worst_rel = 0.0
    acc_a, acc_b = [], []
    for idx in range(len(p.cores)):
        col0 = got[idx]
        ref = expected[idx // geo.s]
        worst_rel = max(worst_rel, ((col0 - ref).abs() / ref.abs()).max().item())
        acc_a.append(col0)
        acc_b.append(ref)
    a = torch.cat(acc_a)
    b = torch.cat(acc_b)
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return pcc, worst_rel, got


def geometry(geo: Geo, force_flat: bool):
    num_owners = 1 if force_flat else min(geo.s, geo.B)
    assert geo.B % num_owners == 0, f"B={geo.B} must divide by num_owners={num_owners}"
    return num_owners, geo.B // num_owners


def build_program(device, p: Plan, x, out, *, variant: str, compute_config):
    geo = p.geo
    paging, combine_impl, gather_chunk, force_flat, split = VARIANTS[variant]
    num_owners, own_rows = geometry(geo, force_flat)
    assert paging == PAGING_ROWMAJOR or combine_impl == COMBINE_RAW, "coalesced landing needs the strided combine"

    # output o -> first tile index, and the step between o's successive partials.
    if paging == PAGING_ROWMAJOR:
        row_pitch, reduce_stride = geo.s, 1
    else:
        row_pitch, reduce_stride = 1, own_rows

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
        # Landing buffer for the owners' finalized rows on the root.  Sized to the
        # WHOLE run so no block reuses another's pages — the bench measures the
        # gather, not a reuse protocol.
        _cb(CB_BCAST_STAGE, geo.nb * geo.B, stat_tile, ttnn.float32),
    ]

    cfg = ttnn.McastConfig(
        noc=ttnn.NOC.NOC_0,
        handshake=False,
        sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED],
    )
    mcast_by_group = {}
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
    assert len(mcast_ct) == 5

    reader_ct = list(mcast_ct) + [
        geo.S,
        geo.B,
        geo.s,
        own_rows,
        num_owners,
        stat_tile,
        SEM_GATHER,
        SEM_STAT_READY,
        geo.shard_rows * geo.S,  # IN_WAIT_TILES — the whole resident shard
    ]
    writer_ct = [
        geo.B,
        geo.s,
        own_rows,
        num_owners,
        stat_tile,
        SEM_GATHER,
        paging,
        gather_chunk,
        split,
    ]
    compute_ct = [
        geo.S,
        geo.B,
        geo.s,
        own_rows,
        geo.shard_rows * geo.S,
        combine_impl,
        row_pitch,
        reduce_stride,
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    inv_w_bits = _f32_bits(1.0 / float(geo.width))
    eps_bits = _f32_bits(EPS)

    for gi, g in enumerate(p.groups):
        root_x, root_y = g["root_virtual"]
        mc = mcast_by_group[gi]
        owner_coords = []
        for o in range(num_owners):
            owner_coords.extend(g["virtual"][o])
        for slice_index, (cx, cy) in enumerate(g["cores"]):
            is_root = 1 if slice_index == 0 else 0
            is_owner = 1 if slice_index < num_owners else 0
            reader_rt[cx][cy] = list(mc.runtime_args(ttnn.CoreCoord(cx, cy))) + [
                geo.nb,
                is_root,
                is_owner,
                slice_index * own_rows,
                root_x,
                root_y,
            ]
            writer_rt[cx][cy] = [geo.nb, slice_index] + owner_coords
            compute_rt[cx][cy] = [geo.nb, is_owner, inv_w_bits, eps_bits]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gc_reader.cpp"),
            core_ranges=p.grid,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gc_writer.cpp"),
            core_ranges=p.grid,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gc_compute.cpp"),
            core_ranges=p.grid,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=compute_config,
        ),
    ]
    semaphores = [
        ttnn.SemaphoreDescriptor(id=i, core_ranges=p.grid, initial_value=0)
        for i in (SEM_MCAST_READY, SEM_MCAST_CONSUMED, SEM_GATHER, 3, SEM_STAT_READY)
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
