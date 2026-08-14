# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: WHERE does rms_norm's `apply_gamma` pass belong?

WHAT IS ISOLATED
    One row-group of `s` cores splits a hidden axis of `s*S` tiles.  Each core holds
    a resident L1 input shard of `nb*B` tile-rows x `S` tiles and a resident L1
    output shard of the same shape, plus a DRAM `gamma` of [W].  The bench runs the
    op's whole per-core pipeline for that geometry:

        Sum(x^2) -> cross-core combine (reduce-scatter, the shipped topology)
                 -> x *= 1/rms -> x *= gamma -> store

    Held identical in every variant (NOT the concept under test): the reader, the
    writer, the gather/broadcast topology, the reduce helper + finalize, the DRAM
    gamma load and its deferred barrier, every CB size, and the precision contract.
    The ONLY thing that moves is WHERE the gamma multiply happens.

WHY THE COMBINE HAS TO BE IN THE BENCH
    The idea is not "make the gamma multiply cheaper" -- the multiply costs the same
    wherever it runs.  It is "move it off the serial post-combine tail into the
    window where the compute thread is already idle waiting for the cross-core
    combine".  A local-compute-only microbench cannot see that; it would report a
    dead flat NULL for a change that is worth ~10% of the wall.  So the combine is
    reconstructed, not stubbed.

VARIANTS
    baseline      sumsq -> [combine] -> x*=rms (in place) -> x*=gamma (-> out shard)
                  == the shipped op (rms_norm_compute.cpp, GAMMA_FUSED=0).
    gamma_first   sumsq -> x*=gamma (in place) -> [combine] -> x*=rms (-> out shard)
                  Same three passes, same helpers, same tile/pack counts.  The gamma
                  pass now runs while the gather+reduce+broadcast is in flight.
    fused         [sumsq AND x*gamma in ONE pass] -> [combine] -> x*=rms (-> out)
                  Two passes.  RAW LLK (see gis_compute.cpp): the chain cannot emit
                  a per-tile pack from a DEST-accumulating walk.

    `_db<N>` suffix on any variant overrides DEST_BLOCK_TILES (the tiles-per-DEST-
    window knob on the streaming eltwise passes; the op ships 8).

PRECISION CONTRACT -- FIXED, not a lever: bf16 x / out / gamma, float32 stat tiles,
math_fidelity=HiFi2, fp32_dest_acc_en=False, math_approx_mode=False.  Every variant
runs under the identical config.  The reorder changes WHICH product is rounded to
bf16 first -- (x*rms)*gamma vs (x*gamma)*rms -- and that is measured, not assumed
(see the pcc column): 0.999909 vs 0.999908 on the focus case, i.e. PCC-neutral.

MEASURED -- Blackhole p150b @1350 MHz, device kernel ns, one fresh run per point.
The `baseline` column is the shipped op's order and reproduces the real op's own
measured 34616 ns on the focus shape to 0.8%, so this is a faithful reconstruction.
Repeats land within 0.1-0.4%, which is this bench's noise floor.

    case              geometry                baseline  gamma_first        fused
    FOCUS         s8 S4 B16 nb2  W=1024          34350  30579 1.123x  38634 0.889x
    S8            s8 S8 B16 nb2  W=2048          44426  37447 1.186x  INEXPRESSIBLE
    S16           s4 S16 B8 nb2  W=2048          36061  33496 1.077x  INEXPRESSIBLE
    B1            s8 S4 B1  nb8  W=1024          30461  27587 1.104x  30709 0.992x
    B8            s8 S4 B8  nb4  W=1024          38074  33739 1.128x  41860 0.910x
    B32           s8 S4 B32 nb1  W=1024          32590  29012 1.123x  37216 0.876x
    s2            s2 S4 B16 nb2  W=256           42104  38407 1.096x  46639 0.903x
    s16           s16 S4 B16 nb2 W=2048          40614  36782 1.104x  44824 0.906x
    s1  NO COMBINE s1 S4 B16 nb2 W=128           45608  46632 0.978x  51550 0.884x
    focus_nogamma s8 S4 B16 nb2  W=1024          30193  30138 1.002x  30170 1.001x
    s1_gl1     (s1, gamma in L1, control)        45612  45550 1.001x            -
    focus_gl1  (focus, gamma in L1, control)     34267  30630 1.119x            -

Four things the table says:

 1. Moving the gamma multiply IN FRONT of the cross-core combine is worth 1.08-1.19x
    of the WHOLE op wall wherever the hidden axis is split (s > 1).  The saving on
    the focus case is 3771 ns, which is the `cp_apply_gamma` stage cost measured on
    the UNPACK thread (3564 ns/core) -- the pass did not get cheaper, it stopped
    being on the serial post-combine tail and now runs inside the window this thread
    spent blocked in `cp_rms_wait`.

 2. FUSING it into the sum-of-squares pass (two passes instead of three) is a
    REGRESSION of 0.88-0.99x everywhere.  The fold removes no work -- unfused,
    x is unpacked 4x and packed 2x per tile across three passes; fused, 4x and 2x
    across two -- so all it changes is WHEN the row's Sum(x*x) partial exists.  The
    fused pass doubles the work per tile-row, so the last partial lands ~2x later,
    the gather cannot start, and the whole group's combine slips.  The premise
    "removes a whole read+pack per block" does not hold: `sum_of_squares` packs
    nothing today, so folding gamma in ADDS that pack back into pass 1.

 3. The s == 1 (no cross-core combine) regression is NOT the pattern.  With gamma
    resident in L1 the same reorder is dead flat there (45550 vs 45612, +0.1%),
    while the focus win is unchanged (1.119x).  What regresses at s == 1 is gamma's
    DRAM ARRIVAL: the op defers gamma's read barrier because the combine hides its
    ~1 us round trip, and at s == 1 there is no combine to hide it behind, so moving
    the consumer earlier un-hides it (the 1024 ns delta is that round trip).

 4. No gamma -> nothing to move -> flat to 0.2%.  That is IN the domain, not an
    exception.

DEST_BLOCK_TILES (the secondary sweep; the op ships 8, the kernel clamps it to the
largest divisor of S).  The claim under test was "8 -> 4 is a free ~4% at S=8/16".
It is REFUTED -- 8 is the best value at every S measured:

    case        db8 (shipped)   db4        db2        db1        db16
    FOCUS S=4   34321           34460      35295      37789      34437
                                0.996x     0.972x     0.908x     0.997x
    S8          44407           44953      46556      51660      -
                                0.988x     0.954x     0.860x
    S16         36061           36482      37927      -          36232
                                0.988x     0.951x                0.995x
    S8 + gamma_first
                37447           37965      -          -          -
                                0.986x

At S=4 the knob is inert (8, 4 and 16 all clamp to the same window of 4, and the
three land within 0.4% == noise).  At S=8/16, dropping to 4 costs a repeatable
1.2%, and the loss grows monotonically as the window shrinks.  16 is also inert
(the chain clamps block_size to DEST_AUTO_LIMIT = 8 at runtime).
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

# CB slots — the same numbering the real op uses, so the kernels read the same.
CB_IN = 0
CB_GAMMA = 1
CB_SQ_PARTIALS = 2
CB_SLICE_STAT = 3
CB_GATHERED = 4
CB_RMS_BCAST = 5
CB_RMS_RECIP = 6
CB_SCALER = 7
CB_OUT = 9
CB_THREAD_SYNC = 12

SEM_MCAST_READY = 0
SEM_MCAST_CONSUMED = 1
SEM_GATHER = 2
SEM_STAT_READY = 3

V_BASELINE = 0
V_GAMMA_FIRST = 1
V_FUSED = 2

DEFAULT_DEST_BLOCK = 8  # rms_norm_program_descriptor.DEST_BLOCK_TILES

# name -> (variant code, dest_block_tiles)
VARIANTS = {
    "baseline": (V_BASELINE, DEFAULT_DEST_BLOCK),
    "gamma_first": (V_GAMMA_FIRST, DEFAULT_DEST_BLOCK),
    "fused": (V_FUSED, DEFAULT_DEST_BLOCK),
    # DEST_BLOCK_TILES sweep (secondary, same stage)
    "baseline_db1": (V_BASELINE, 1),
    "baseline_db2": (V_BASELINE, 2),
    "baseline_db4": (V_BASELINE, 4),
    "baseline_db16": (V_BASELINE, 16),
    "gamma_first_db4": (V_GAMMA_FIRST, 4),
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
    has_gamma: bool = True
    # Control knob for the s==1 exception: put gamma in L1 instead of DRAM, which
    # removes its ~1 us round trip from the comparison entirely.  Diagnostic only —
    # the op's gamma is a DRAM tensor.
    gamma_in_l1: bool = False

    @property
    def shard_rows(self) -> int:
        return self.nb * self.B

    @property
    def width(self) -> int:
        return self.s * self.S * TILE

    @property
    def label(self) -> str:
        g = ("gl1" if self.gamma_in_l1 else "g") if self.has_gamma else "nog"
        return f"s{self.s}_S{self.S}_B{self.B}_nb{self.nb}_{g}"


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
        assert span == geo.s, f"row-group {r} {gcores} is not a rectangle (bbox holds {span}, s={geo.s})"
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


def _shard_mc(p: Plan):
    geo = p.geo
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(p.grid, [geo.shard_rows * TILE, geo.S * TILE], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _x_torch(p: Plan, seed: int):
    """(per-group x, the flattened per-core band layout the L1 shard wants)."""
    geo = p.geo
    rows = geo.shard_rows * TILE
    torch.manual_seed(seed)
    x_groups = torch.randn(len(p.groups), rows, geo.width, dtype=torch.float32).to(torch.bfloat16)
    slice_w = geo.S * TILE
    bands = []
    for idx in range(len(p.cores)):
        r, c = divmod(idx, geo.s)
        bands.append(x_groups[r][:, c * slice_w : (c + 1) * slice_w])
    flat = torch.cat(bands, dim=0).reshape(1, 1, len(p.cores) * rows, slice_w)
    return x_groups, flat


def make_input(device, p: Plan, *, seed: int = 42):
    """A FRESH resident input shard — every variant rewrites x in place."""
    _g, flat = _x_torch(p, seed)
    return ttnn.from_torch(
        flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_shard_mc(p)
    )


def make_tensors(device, p: Plan, *, seed: int = 42):
    """Resident L1 in/out shards + a DRAM gamma + the fp32 torch reference."""
    geo = p.geo
    rows = geo.shard_rows * TILE
    slice_w = geo.S * TILE
    x_groups, _flat = _x_torch(p, seed)

    x = make_input(device, p, seed=seed)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, len(p.cores) * rows, slice_w]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, _shard_mc(p)
    )

    if geo.has_gamma:
        g_t = (torch.rand(geo.width, dtype=torch.float32) * 1.5 + 0.25).to(torch.bfloat16)
        gamma = ttnn.from_torch(
            g_t.reshape(1, 1, 1, geo.width),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG if geo.gamma_in_l1 else ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        g_t = torch.ones(geo.width, dtype=torch.bfloat16)
        gamma = None

    xf = x_groups.to(torch.float32)
    rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)  # (ngroups, rows, 1)
    expected = xf * rms * g_t.to(torch.float32)  # (ngroups, rows, width)
    return x, out, gamma, expected


def check(p: Plan, out, expected):
    """Every core's output shard must hold its own band of x * 1/rms * gamma."""
    geo = p.geo
    rows = geo.shard_rows * TILE
    slice_w = geo.S * TILE
    got = ttnn.to_torch(out).to(torch.float32).reshape(-1, slice_w)
    acc_a, acc_b = [], []
    worst_rel = 0.0
    for idx in range(len(p.cores)):
        r, c = divmod(idx, geo.s)
        band = got[idx * rows : (idx + 1) * rows, :]
        ref = expected[r][:, c * slice_w : (c + 1) * slice_w]
        denom = ref.abs().clamp_min(1e-3)
        worst_rel = max(worst_rel, ((band - ref).abs() / denom).max().item())
        acc_a.append(band.reshape(-1))
        acc_b.append(ref.reshape(-1))
    a = torch.cat(acc_a)
    b = torch.cat(acc_b)
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return pcc, worst_rel


def build_program(device, p: Plan, x, out, gamma, *, variant: str, compute_config):
    geo = p.geo
    vcode, dest_block = VARIANTS[variant]
    has_gamma = 1 if geo.has_gamma else 0
    if not has_gamma:
        vcode = V_BASELINE  # nothing to fold; every variant is the same program

    num_owners = min(geo.s, geo.B)
    assert geo.B % num_owners == 0
    own_rows = geo.B // num_owners

    bf16_tile = ttnn.tile_size(ttnn.bfloat16)
    stat_tile = ttnn.tile_size(ttnn.float32)

    def _cb(index, pages, page_size, dtype):
        return ttnn.CBDescriptor(
            total_size=pages * page_size,
            core_ranges=p.grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, x),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
        _cb(CB_SQ_PARTIALS, geo.B, stat_tile, ttnn.float32),
        _cb(CB_RMS_RECIP, geo.B, stat_tile, ttnn.float32),
        _cb(CB_SCALER, 1, bf16_tile, ttnn.bfloat16),
        _cb(CB_THREAD_SYNC, 1, 16, ttnn.bfloat16),
    ]
    if has_gamma:
        cbs.append(_cb(CB_GAMMA, geo.S, bf16_tile, ttnn.bfloat16))
    if geo.s > 1:
        cbs.append(_cb(CB_GATHERED, geo.s * own_rows, stat_tile, ttnn.float32))
        cbs.append(_cb(CB_RMS_BCAST, geo.B, stat_tile, ttnn.float32))
        if num_owners > 1:
            cbs.append(_cb(CB_SLICE_STAT, own_rows, stat_tile, ttnn.float32))

    # ---- mcast wire, one per row-group (identical CT) ----
    mcast_by_group = {}
    if geo.s > 1:
        cfg = ttnn.McastConfig(
            noc=ttnn.NOC.NOC_0,
            handshake=(geo.nb > 1),  # cb_rms_recip is B pages and is popped every block
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

    in_wait_tiles = geo.shard_rows * geo.S

    reader_ct = (
        list(mcast_ct)
        + [
            geo.S,
            geo.B,
            geo.s,
            has_gamma,
            bf16_tile,  # GAMMA_TILE_BYTES
            2,  # GAMMA_ELEM_BYTES
            SEM_GATHER,
            SEM_STAT_READY,
            stat_tile,
            in_wait_tiles,
            num_owners,
            own_rows,
        ]
        + list(
            ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
            if has_gamma
            else ttnn.TensorAccessorArgs().get_compile_time_args()
        )
    )
    writer_ct = [
        geo.S,
        geo.B,
        geo.s,
        stat_tile,
        SEM_GATHER,
        num_owners,
        own_rows,
    ]
    compute_ct = [
        geo.S,
        geo.B,
        geo.s,
        has_gamma,
        in_wait_tiles,  # IN_WAIT_TILES
        in_wait_tiles,  # IN_CAPACITY_TILES
        dest_block,
        num_owners,
        own_rows,
        vcode,
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    inv_w_bits = _f32_bits(1.0 / float(geo.width))
    eps_bits = _f32_bits(EPS)
    gamma_addr = gamma.buffer_address() if has_gamma else 0

    for gi, g in enumerate(p.groups):
        vxlo, vylo, vxhi, vyhi = g["bbox_virtual"]
        root_x, root_y = g["root_virtual"]
        mc = mcast_by_group.get(gi)
        owner_coords = []
        for o in range(num_owners):
            owner_coords.extend(g["virtual"][o])
        for slice_index, (cx, cy) in enumerate(g["cores"]):
            is_root = 1 if slice_index == 0 else 0
            is_owner = 1 if slice_index < num_owners else 0
            mcast_rt = list(mc.runtime_args(ttnn.CoreCoord(cx, cy))) if mc is not None else [0, 0, 0, 0]
            reader_rt[cx][cy] = mcast_rt + [
                geo.nb,
                is_root,
                is_owner,
                slice_index * own_rows,
                slice_index * geo.S,  # gamma slice base, in tiles
                gamma_addr,
                root_x,
                root_y,
            ]
            writer_rt[cx][cy] = [geo.nb, slice_index] + owner_coords
            compute_rt[cx][cy] = [geo.nb, is_owner, inv_w_bits, eps_bits]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gis_reader.cpp"),
            core_ranges=p.grid,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gis_writer.cpp"),
            core_ranges=p.grid,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gis_compute.cpp"),
            core_ranges=p.grid,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=compute_config,
        ),
    ]
    semaphores = [
        ttnn.SemaphoreDescriptor(id=i, core_ranges=p.grid, initial_value=0)
        for i in (SEM_MCAST_READY, SEM_MCAST_CONSUMED, SEM_GATHER, SEM_STAT_READY)
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


def target_compute_config():
    """The perf group's FIXED precision contract — identical for every variant."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=False,
        math_approx_mode=False,
    )


def run_variant(device, p: Plan, x, out, gamma, *, variant: str):
    pd = build_program(device, p, x, out, gamma, variant=variant, compute_config=target_compute_config())
    # `out` must come LAST: generic_op returns the final tensor of the io list.
    tensors = [x] + ([gamma] if gamma is not None else []) + [out]
    return ttnn.generic_op(tensors, pd)
