# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off harness for rms_norm's CROSS-CORE STAT COMBINE.

Reconstruction, not a copy: the only thing this program does is
`sum(x^2) -> gather -> root combine + finalize -> broadcast`.  x is a resident
L1 shard, there is no gamma, and nothing scales or stores x.  The measured
delta between MODEs is therefore attributable to the combine alone.

Four MODEs (see csg_compute.cpp for the full statement of each):

    0 MODE_RAW_TILE     the op's CURRENT approach == the honest baseline
    1 MODE_COLLAPSE_4K  collapse on the contributor, still ship 4 KB
    2 MODE_COLLAPSE_2K  collapse, ship the 2 valid faces (2 KB)
    3 MODE_ROW_128B     collapse + in-DEST transpose, ship 2 x 64 B into ROW c
                        of ONE landing tile; root does REDUCE_COL + transpose back
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

CB_INPUT_TILES = 0
CB_SQ_PARTIALS = 2
CB_SLICE_STAT = 3
CB_GATHERED_PARTIALS = 4
CB_RMS_BCAST = 5
CB_RMS_RECIP = 6
CB_SCALER = 7

SEM_MCAST_READY = 0
SEM_MCAST_CONSUMED = 1
SEM_GATHER_PROGRESS = 2
# Orders the ROOT's boot-time zeroing of the landing buffer against the
# CONTRIBUTORS' first gather write into it (different cores, same L1 — nothing
# else orders them; see the reader's comment).
SEM_LANDING_READY = 3

MODE_RAW_TILE = 0
MODE_COLLAPSE_4K = 1
MODE_COLLAPSE_2K = 2
MODE_ROW_128B = 3

MODE_NAMES = {
    MODE_RAW_TILE: "baseline_raw_4k",
    MODE_COLLAPSE_4K: "collapse_4k",
    MODE_COLLAPSE_2K: "collapse_2k_faces",
    MODE_ROW_128B: "row_128b_transposed",
}


def _div_up(a, b):
    return (a + b - 1) // b


def _f32_bits(v):
    return struct.unpack("I", struct.pack("f", float(v)))[0]


def plan(input_tensor, *, block_rows=None):
    """Read (s, S, shard_rows, groups) off the shard spec, exactly as the op does."""
    spec = input_tensor.memory_config().shard_spec
    assert spec is not None, "compact_stat_gather bench requires a SHARDED, TILE-layout input"
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
    return {
        "cores": cores,
        "s": num_slices,
        "S": slice_tiles,
        "B": b,
        "shard_rows": shard_rows,
        "groups": num_groups,
        "row_tiles": row_tiles,
    }


def landing_pages(mode, s, b):
    """Root landing-buffer pages per block."""
    rows = _div_up(s, TILE_DIM) if mode == MODE_ROW_128B else s
    return rows, rows * b


def cb_bytes(mode, s, S, b, *, in_tile_bytes, stat_tile=4096):
    """Per-core CB bytes EXCLUDING the resident input shard (bound zero-copy)."""
    _rows, gather = landing_pages(mode, s, b)
    total = b * stat_tile  # cb_sq_partials
    if mode != MODE_RAW_TILE:
        total += b * stat_tile  # cb_slice_stat
    total += gather * stat_tile  # cb_gathered_partials
    total += b * stat_tile  # cb_rms_bcast
    total += b * stat_tile  # cb_rms_recip
    total += ttnn.tile_size(ttnn.bfloat16)  # cb_scaler
    return total


def create_program_descriptor(
    input_tensor,
    stat_output,
    *,
    mode,
    epsilon=1e-6,
    block_rows=None,
    poison_landing=False,
    drain=True,
    compute_kernel_config=None,
):
    device = input_tensor.device()
    p = plan(input_tensor, block_rows=block_rows)
    cores, s, S, B = p["cores"], p["s"], p["S"], p["B"]
    shard_rows, groups = p["shard_rows"], p["groups"]

    W = int(input_tensor.shape[-1])
    in_tile = ttnn.tile_size(input_tensor.dtype)
    stat_tile = ttnn.tile_size(ttnn.float32)
    bf16_tile = ttnn.tile_size(ttnn.bfloat16)
    landing_rows, gather_pages = landing_pages(mode, s, B)

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
    if mode != MODE_RAW_TILE:
        cbs.append(_cb(CB_SLICE_STAT, B, stat_tile, ttnn.float32))

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
        1 if poison_landing else 0,
        SEM_LANDING_READY,
    ]
    writer_ct = [S, B, s, stat_tile, SEM_GATHER_PROGRESS, mode, SEM_LANDING_READY, 1 if drain else 0]
    writer_ct.extend(ttnn.TensorAccessorArgs(stat_output).get_compile_time_args())
    compute_ct = [S, B, s, mode, in_wait_tiles, landing_rows]

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
        mc = mcast_by_group[idx]
        for slice_index, (cx, cy) in enumerate(g["cores"]):
            is_root = 1 if (cx, cy) == (ox, oy) else 0
            reader_rt[cx][cy] = (
                list(mc.runtime_args(ttnn.CoreCoord(cx, cy)))
                + [
                    g["num_blocks"],
                    is_root,
                ]
                + rect
            )
            writer_rt[cx][cy] = [
                out_addr,
                g["num_blocks"],
                slice_index,
                root_virt.x,
                root_virt.y,
                is_root,
                g["page_base"],
            ]
            compute_rt[cx][cy] = [g["num_blocks"], is_root, inv_w_bits, eps_bits]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "csg_reader.cpp"),
            core_ranges=kernel_cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "csg_writer.cpp"),
            core_ranges=kernel_cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "csg_compute.cpp"),
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
        ttnn.SemaphoreDescriptor(id=SEM_LANDING_READY, core_ranges=kernel_cores, initial_value=0),
    ]

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs), p
