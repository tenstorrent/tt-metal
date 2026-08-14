# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off for rms_norm's CROSS-CORE STAT GATHER — round 2.

Round 1's `compact_stat_gather` bench measured this idea against the FLAT-ROOT
combine.  That combine no longer exists: Perf 1 graduated the reduce-scatter
(I1), so the honest baseline is now

    num_owners = largest divisor of block_rows that is <= min(s, block_rows)
    own_rows   = block_rows / num_owners
    contributor c ships, per block row r, a WHOLE 4 KB fp32 tile to owner r//own_rows
                  at gather page (r % own_rows) * s + c
    owner o     runs ONE reduce<SUM, REDUCE_ROW> over the (own_rows, s) gathered
                  block with the finalize fused in
    owner o     funnels its own_rows finished tiles to the root's cb_rms_bcast
    root        broadcasts the block's block_rows finalized 1/rms tiles

This file reconstructs exactly that, plus three alternative gather spellings.
Everything downstream of the combine (scale x, gamma, store x) is absent, so the
measured delta is attributable to the gather + owner combine alone.

MODES
    0 raw_4k        the op's CURRENT approach == the honest baseline
    1 row_128b      collapse + transpose_dest at the CONTRIBUTOR, ship 2 x 64 B
                    into ROW c of the owner's per-owned-row landing tile; owner
                    does REDUCE_COL + transpose back + finalize
    2 collapse_2k   collapse at the contributor (no transpose), ship the two
                    column-0-bearing faces (2 KB) into the baseline landing;
                    owner code identical to the baseline.  A CONTROL that
                    separates "fewer bytes" from "fewer tiles".
    3 row_64b_probe ABLATION PROBE, NUMERICALLY WRONG BY CONSTRUCTION: mode 1
                    with only the first of its two 64 B writes issued.  Exists
                    solely to price the second NoC transaction; never an option.

PRECISION CONTRACT (fixed, never a lever): bf16 activations, float32 stat tiles,
math_fidelity=HiFi2, fp32_dest_acc_en=False, math_approx_mode=False.  Identical
for every mode.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

CB_INPUT_TILES = 0
CB_SQ_PARTIALS = 2
CB_SLICE_STAT = 3  # the OWNER's reduce output (funnelled to the root)
CB_GATHERED_PARTIALS = 4
CB_RMS_BCAST = 5
CB_RMS_RECIP = 6
CB_SCALER = 7
CB_STAT_COMPACT = 10  # the CONTRIBUTOR's collapsed staging (modes 1/2/3)
CB_ZEROS = 11  # one fp32 tile of zeros: the boot pad-zero source (modes 1/2/3)

SEM_MCAST_READY = 0
SEM_MCAST_CONSUMED = 1
SEM_GATHER_PROGRESS = 2
SEM_STAT_READY = 3  # owner -> root funnel completion

MODE_RAW_4K = 0
MODE_ROW_128B = 1
MODE_COLLAPSE_2K = 2
MODE_ROW_64B_PROBE = 3

MODE_NAMES = {
    MODE_RAW_4K: "baseline_raw_4k",
    MODE_ROW_128B: "row_128b",
    MODE_COLLAPSE_2K: "collapse_2k",
    MODE_ROW_64B_PROBE: "row_64b_probe",
}


def _div_up(a, b):
    return (a + b - 1) // b


def _f32_bits(v):
    return struct.unpack("I", struct.pack("f", float(v)))[0]


def combine_owners(num_slices: int, block_rows: int) -> int:
    """Verbatim `_combine_owners` from the op's program descriptor."""
    cap = min(num_slices, block_rows)
    for d in range(cap, 0, -1):
        if block_rows % d == 0:
            return d
    return 1


def plan(input_tensor, *, block_rows=None):
    """Read (s, S, shard_rows, groups) off the shard spec, exactly as the op does."""
    spec = input_tensor.memory_config().shard_spec
    assert spec is not None, "this bench requires a SHARDED, TILE-layout input"
    row_wise = spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    cores = [(int(c.x), int(c.y)) for c in ttnn.corerange_to_cores(spec.grid, None, row_wise)]
    shard_h, shard_w = int(spec.shape[0]), int(spec.shape[1])
    assert shard_h % TILE_DIM == 0 and shard_w % TILE_DIM == 0

    padded = list(input_tensor.padded_shape)
    row_tiles = 1
    for d in padded[:-2]:
        row_tiles *= d
    row_tiles *= _div_up(padded[-2], TILE_DIM)
    hidden_tiles = _div_up(padded[-1], TILE_DIM)

    slice_tiles = shard_w // TILE_DIM
    shard_rows = shard_h // TILE_DIM
    num_slices = _div_up(hidden_tiles, slice_tiles)
    num_groups = _div_up(row_tiles, shard_rows)
    assert num_groups * num_slices == len(
        cores
    ), f"shard grid holds {len(cores)} cores but shape implies {num_groups} x {num_slices}"

    b = block_rows if block_rows is not None else shard_rows
    assert shard_rows % b == 0, f"block_rows {b} must divide shard_rows {shard_rows}"
    num_owners = combine_owners(num_slices, b)
    return {
        "cores": cores,
        "s": num_slices,
        "S": slice_tiles,
        "B": b,
        "shard_rows": shard_rows,
        "groups": num_groups,
        "row_tiles": row_tiles,
        "num_owners": num_owners,
        "own_rows": b // num_owners,
    }


def landing_geometry(mode, s, own_rows):
    """(landing_tile_rows, gather_pages) for one owner, per block.

    Baseline / collapse_2k: one landing tile per (owned row, contributor).
    row_128b / probe:       one landing tile per owned row, holding up to 32
                            contributors as ROWS.  ceil(s/32) tiles when a row-
                            group is wider than a tile.
    """
    if mode in (MODE_ROW_128B, MODE_ROW_64B_PROBE):
        rows = _div_up(s, TILE_DIM)
        return rows, rows * own_rows
    return s, s * own_rows


def cb_bytes(mode, s, S, b, num_owners, own_rows, *, stat_tile=4096):
    """Per-core CB bytes EXCLUDING the resident input shard (bound zero-copy)."""
    _rows, gather = landing_geometry(mode, s, own_rows)
    total = b * stat_tile  # cb_sq_partials
    total += gather * stat_tile  # cb_gathered_partials
    total += b * stat_tile  # cb_rms_bcast
    total += b * stat_tile  # cb_rms_recip
    if num_owners > 1:
        total += own_rows * stat_tile  # cb_slice_stat
    if mode != MODE_RAW_4K:
        total += b * stat_tile  # cb_stat_compact
        total += stat_tile  # cb_zeros
    total += ttnn.tile_size(ttnn.bfloat16)  # cb_scaler
    return total


def create_program_descriptor(
    input_tensor,
    stat_output,
    *,
    mode,
    epsilon=1e-6,
    block_rows=None,
    drain=True,
    compute_kernel_config=None,
):
    device = input_tensor.device()
    p = plan(input_tensor, block_rows=block_rows)
    cores, s, S, B = p["cores"], p["s"], p["S"], p["B"]
    shard_rows, groups = p["shard_rows"], p["groups"]
    num_owners, own_rows = p["num_owners"], p["own_rows"]

    W = int(input_tensor.shape[-1])
    stat_tile = ttnn.tile_size(ttnn.float32)
    bf16_tile = ttnn.tile_size(ttnn.bfloat16)
    landing_rows, gather_pages = landing_geometry(mode, s, own_rows)

    # ---- groups: `groups` consecutive runs of s cores, in shard order ----
    row_tiles = p["row_tiles"]
    gdefs = []
    for r in range(groups):
        gcores = cores[r * s : (r + 1) * s]
        rows = max(0, min(shard_rows, row_tiles - r * shard_rows))
        assert rows == shard_rows, "bench requires a whole shard per group (no ragged tail)"
        gdefs.append({"origin": gcores[0], "cores": gcores, "num_blocks": rows // B, "page_base": r * shard_rows})

    def _bbox(cs):
        xs = [c[0] for c in cs]
        ys = [c[1] for c in cs]
        return ttnn.CoreRange(ttnn.CoreCoord(min(xs), min(ys)), ttnn.CoreCoord(max(xs), max(ys)))

    kernel_cores = ttnn.CoreRangeSet([_bbox(g["cores"]) for g in gdefs])

    single_shot = all(g["num_blocks"] <= 1 for g in gdefs)
    cfg = ttnn.McastConfig(
        noc=ttnn.NOC.NOC_0,
        handshake=not single_shot,
        sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED],
    )
    mcast_by_group = {}
    for idx, g in enumerate(gdefs):
        ox, oy = g["origin"]
        mcast_by_group[idx] = ttnn.Mcast2D(
            device, ttnn.CoreRangeSet([_bbox(g["cores"])]), ttnn.CoreCoord(ox, oy), cfg, s - 1
        )
    mcast_ct = list(mcast_by_group[0].compile_time_args())

    def _cb(index, pages, page_size, dtype):
        return ttnn.CBDescriptor(
            total_size=pages * page_size,
            core_ranges=kernel_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_TILES, input_tensor),
        _cb(CB_SQ_PARTIALS, B, stat_tile, ttnn.float32),
        _cb(CB_GATHERED_PARTIALS, gather_pages, stat_tile, ttnn.float32),
        _cb(CB_RMS_BCAST, B, stat_tile, ttnn.float32),
        _cb(CB_RMS_RECIP, B, stat_tile, ttnn.float32),
        _cb(CB_SCALER, 1, bf16_tile, ttnn.bfloat16),
    ]
    if num_owners > 1:
        cbs.append(_cb(CB_SLICE_STAT, own_rows, stat_tile, ttnn.float32))
    if mode != MODE_RAW_4K:
        cbs.append(_cb(CB_STAT_COMPACT, B, stat_tile, ttnn.float32))
        cbs.append(_cb(CB_ZEROS, 1, stat_tile, ttnn.float32))

    in_wait_tiles = shard_rows * S

    reader_ct = list(mcast_ct) + [
        S,
        B,
        s,
        SEM_GATHER_PROGRESS,
        stat_tile,
        in_wait_tiles,
        mode,
        landing_rows,
        SEM_STAT_READY,
        num_owners,
        own_rows,
    ]
    writer_ct = [
        S,
        B,
        s,
        stat_tile,
        SEM_GATHER_PROGRESS,
        mode,
        1 if drain else 0,
        num_owners,
        own_rows,
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(stat_output).get_compile_time_args())
    compute_ct = [S, B, s, mode, in_wait_tiles, landing_rows, num_owners, own_rows]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    inv_w_bits = _f32_bits(1.0 / float(W))
    eps_bits = _f32_bits(epsilon)
    out_addr = stat_output.buffer_address()

    for idx, g in enumerate(gdefs):
        ox, oy = g["origin"]
        root_virt = device.worker_core_from_logical_core(ttnn.CoreCoord(ox, oy))
        xs = [c[0] for c in g["cores"]]
        ys = [c[1] for c in g["cores"]]
        v0 = device.worker_core_from_logical_core(ttnn.CoreCoord(min(xs), min(ys)))
        v1 = device.worker_core_from_logical_core(ttnn.CoreCoord(max(xs), max(ys)))
        rect = [min(v0.x, v1.x), min(v0.y, v1.y), max(v0.x, v1.x), max(v0.y, v1.y), len(g["cores"])]
        # The owners are the FIRST num_owners cores of the group in slice order.
        owner_xy = []
        for o in range(num_owners):
            cx, cy = g["cores"][o]
            v = device.worker_core_from_logical_core(ttnn.CoreCoord(cx, cy))
            owner_xy.extend([int(v.x), int(v.y)])
        mc = mcast_by_group[idx]
        for slice_index, (cx, cy) in enumerate(g["cores"]):
            is_root = 1 if (cx, cy) == (ox, oy) else 0
            is_owner = 1 if slice_index < num_owners else 0
            reader_rt[cx][cy] = (
                list(mc.runtime_args(ttnn.CoreCoord(cx, cy)))
                + [
                    g["num_blocks"],
                    is_root,
                    is_owner,
                    slice_index * own_rows,  # my_first_row (owners only)
                    int(root_virt.x),
                    int(root_virt.y),
                ]
                + rect
            )
            writer_rt[cx][cy] = [
                out_addr,
                g["num_blocks"],
                slice_index,
                int(root_virt.x),
                int(root_virt.y),
                is_root,
                g["page_base"],
            ] + owner_xy
            compute_rt[cx][cy] = [g["num_blocks"], is_owner, inv_w_bits, eps_bits]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "csg2_reader.cpp"),
            core_ranges=kernel_cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "csg2_writer.cpp"),
            core_ranges=kernel_cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "csg2_compute.cpp"),
            core_ranges=kernel_cores,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=compute_kernel_config,
        ),
    ]

    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_READY, core_ranges=kernel_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_CONSUMED, core_ranges=kernel_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_GATHER_PROGRESS, core_ranges=kernel_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_STAT_READY, core_ranges=kernel_cores, initial_value=0),
    ]

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs), p
