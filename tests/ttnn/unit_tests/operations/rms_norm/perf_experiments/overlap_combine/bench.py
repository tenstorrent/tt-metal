# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""overlap_combine — an ISOLATED bake-off of the cross-core combine's SCHEDULE.

WHAT IS UNDER TEST
------------------
On a hidden-split sharded plan every core must wait for a row-group root to
gather `s` partial statistics, reduce them, finalize and broadcast 1/rms before
it can scale its own rows.  The op's current structure makes that wait FULLY
EXPOSED: per block a core does Sum(x^2), ships it, and then idles until 1/rms
comes back.  Two structural levers are measured here:

  1. PIPELINE — software-pipeline the block loop.  `Sum(x^2)` for block b+1 has
     no unmet dependency (x is a RESIDENT L1 shard), so it can run — and its stat
     can reach the root — while block b's combine round trip is in flight.
  2. STAT_ROWS > APPLY_ROWS — decouple the combine granularity from the apply
     granularity: ONE coarse round trip per STAT_ROWS tile-rows, while the apply
     pass keeps its (L1/DEST-sized) APPLY_ROWS block.

WHAT IS HELD CONSTANT (isolation)
---------------------------------
  * x and out are resident L1 shards, bound zero-copy to the CBs — there is no
    DRAM read of x at all, which is what makes the combine the whole wall.
  * no gamma, no W-mask, no ROW_MAJOR/tilize, no interleaved path.  Dropping
    gamma removes the op's only DRAM tensor (a separate concern, owned
    elsewhere); it is absent from EVERY variant, so the schedule delta is clean.
    The absolute ns here are therefore below the full op's by the gamma stage.
  * the precision contract is FIXED and identical in every variant:
    bf16 activations, float32 stat tiles, math_fidelity=HiFi2,
    fp32_dest_acc_en=False, math_approx_mode=False.

The reader and writer kernels are IDENTICAL across variants (the writer only
windows the landing buffer when GATHER_DEPTH == 2).  The schedule lives entirely
in the compute kernel's loop order plus the host CB depths.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

# CB indices — same numbering as the op, so the zone names line up.
CB_INPUT_TILES = 0
CB_SQ_PARTIALS = 2
CB_GATHERED_PARTIALS = 4
CB_RMS_BCAST = 5
CB_RMS_RECIP = 6
CB_SCALER = 7
CB_OUTPUT_TILES = 9

SEM_MCAST_READY = 0
SEM_MCAST_CONSUMED = 1
SEM_GATHER_PROGRESS = 2

# Tiles per DEST window on the apply pass — the op's shipped value.  Held constant.
DEST_BLOCK_TILES = 8


def _f32_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", float(value)))[0]


def _shard_core_list(spec):
    row_wise = spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    return [(int(c.x), int(c.y)) for c in ttnn.corerange_to_cores(spec.grid, None, row_wise)]


def plan(input_tensor):
    """Read the row-group partition off the shard spec (the op's `_plan_sharded`)."""
    spec = input_tensor.memory_config().shard_spec
    assert spec is not None, "overlap_combine: needs a sharded input"
    shard_h, shard_w = int(spec.shape[0]), int(spec.shape[1])
    assert shard_h % TILE_DIM == 0 and shard_w % TILE_DIM == 0, "TILE shard must be 32-aligned"

    padded = list(input_tensor.padded_shape)
    w = padded[-1]
    images = 1
    for d in padded[:-2]:
        images *= d
    row_tiles = images * (padded[-2] // TILE_DIM)
    hidden_tiles = w // TILE_DIM

    shard_rows = shard_h // TILE_DIM
    slice_tiles = shard_w // TILE_DIM
    num_row_groups = row_tiles // shard_rows
    num_slices = hidden_tiles // slice_tiles

    cores = _shard_core_list(spec)
    assert num_row_groups * num_slices == len(cores), (
        f"shard grid holds {len(cores)} cores but [{shard_h},{shard_w}] implies " f"{num_row_groups} x {num_slices}"
    )
    assert row_tiles % shard_rows == 0 and hidden_tiles % slice_tiles == 0, "bench needs an exact shard tiling"
    assert num_slices > 1, "the combine only exists on a hidden-split plan"
    return {
        "row_tiles": row_tiles,
        "hidden_tiles": hidden_tiles,
        "num_row_groups": num_row_groups,
        "num_slices": num_slices,
        "slice_tiles": slice_tiles,
        "shard_rows": shard_rows,
        "cores": cores,
        "w": w,
    }


def l1_report(input_tensor, output_tensor, *, stat_rows, pipeline):
    """Per-core L1 bytes: the resident shards + every CB this variant declares."""
    p = plan(input_tensor)
    in_tile = ttnn.tile_size(input_tensor.dtype)
    out_tile = ttnn.tile_size(output_tensor.dtype)
    stat_tile = ttnn.tile_size(ttnn.float32)
    bf16_tile = ttnn.tile_size(ttnn.bfloat16)
    shard_tiles = p["shard_rows"] * p["slice_tiles"]
    # Mirror `create_program_descriptor`: a one-round-trip geometry has nothing to
    # pipeline, so it keeps depth 1 and the two variants are byte-identical.
    depth = 2 if (pipeline and (p["shard_rows"] // stat_rows) > 1) else 1
    shards = shard_tiles * (in_tile + out_tile)
    cbs = (
        depth * stat_rows * stat_tile  # cb_sq_partials
        + depth * p["num_slices"] * stat_rows * stat_tile  # cb_gathered_partials
        + depth * stat_rows * stat_tile  # cb_rms_bcast
        + depth * stat_rows * stat_tile  # cb_rms_recip
        + bf16_tile  # cb_scaler
    )
    return {"shard_bytes": shards, "cb_bytes": cbs, "total_bytes": shards + cbs}


def create_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    stat_rows: int,
    apply_rows: int,
    pipeline: int,
    epsilon: float = 1e-6,
    compute_kernel_config=None,
):
    device = input_tensor.device()
    p = plan(input_tensor)
    S = p["slice_tiles"]
    s = p["num_slices"]
    G = p["num_row_groups"]
    shard_rows = p["shard_rows"]
    shard_tiles = shard_rows * S
    W = p["w"]

    assert shard_rows % stat_rows == 0, f"stat_rows {stat_rows} must divide shard_rows {shard_rows}"
    assert stat_rows % apply_rows == 0, f"apply_rows {apply_rows} must divide stat_rows {stat_rows}"
    num_stat_blocks = shard_rows // stat_rows

    # A one-round-trip geometry has nothing to pipeline, so the deeper CBs would be
    # pure L1 cost with no schedule change.  Keep them at depth 1 there, which is
    # what makes the "flat, not a regression" check on the decode regime honest.
    effective_pipeline = 1 if (pipeline and num_stat_blocks > 1) else 0
    depth = 2 if effective_pipeline else 1

    stat_tile = ttnn.tile_size(ttnn.float32)
    bf16_tile = ttnn.tile_size(ttnn.bfloat16)

    # ---- row groups: G groups of `s` consecutive shard-order cores ----
    core_list = p["cores"]
    groups = [{"cores": core_list[r * s : (r + 1) * s]} for r in range(G)]
    for g in groups:
        g["origin"] = g["cores"][0]

    def _bbox(cores):
        xs = [c[0] for c in cores]
        ys = [c[1] for c in cores]
        return ttnn.CoreRange(ttnn.CoreCoord(min(xs), min(ys)), ttnn.CoreCoord(max(xs), max(ys)))

    kernel_cores_crs = ttnn.CoreRangeSet([_bbox(g["cores"]) for g in groups])
    cb_cores_crs = kernel_cores_crs

    # ---- mcast wire: one Mcast2D per row-group rect (identical CT for all) ----
    single_shot = num_stat_blocks <= 1
    cfg = ttnn.McastConfig(
        noc=ttnn.NOC.NOC_0,
        handshake=not single_shot,
        sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED],
    )
    mcast_by_group = {}
    for idx, g in enumerate(groups):
        ox, oy = g["origin"]
        mcast_by_group[idx] = ttnn.Mcast2D(
            device, ttnn.CoreRangeSet([_bbox(g["cores"])]), ttnn.CoreCoord(ox, oy), cfg, s - 1
        )
    mcast_ct = list(mcast_by_group[0].compile_time_args())
    assert len(mcast_ct) == 5

    def _cb(index, pages, page_size, dtype):
        return ttnn.CBDescriptor(
            total_size=pages * page_size,
            core_ranges=cb_cores_crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    cbs = [
        # x / out consumed NATIVELY: the CB IS the caller's resident shard.
        ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_TILES, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT_TILES, output_tensor),
        _cb(CB_SQ_PARTIALS, depth * stat_rows, stat_tile, ttnn.float32),
        _cb(CB_GATHERED_PARTIALS, depth * s * stat_rows, stat_tile, ttnn.float32),
        _cb(CB_RMS_BCAST, depth * stat_rows, stat_tile, ttnn.float32),
        _cb(CB_RMS_RECIP, depth * stat_rows, stat_tile, ttnn.float32),
        _cb(CB_SCALER, 1, bf16_tile, ttnn.bfloat16),
    ]

    reader_ct = list(mcast_ct) + [stat_rows, s, stat_tile, SEM_GATHER_PROGRESS, shard_tiles]
    writer_ct = [S, stat_rows, s, stat_tile, SEM_GATHER_PROGRESS, depth, shard_rows]
    compute_ct = [S, stat_rows, apply_rows, s, shard_tiles, DEST_BLOCK_TILES, effective_pipeline]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    inv_w_bits = _f32_bits(1.0 / float(W))
    eps_bits = _f32_bits(epsilon)

    for idx, g in enumerate(groups):
        ox, oy = g["origin"]
        root_virt = device.worker_core_from_logical_core(ttnn.CoreCoord(ox, oy))
        mc = mcast_by_group[idx]
        for slice_index, (cx, cy) in enumerate(g["cores"]):
            is_root = 1 if (cx, cy) == (ox, oy) else 0
            mcast_rt = list(mc.runtime_args(ttnn.CoreCoord(cx, cy)))
            reader_rt[cx][cy] = list(mcast_rt) + [num_stat_blocks, is_root]
            writer_rt[cx][cy] = [num_stat_blocks, root_virt.x, root_virt.y, slice_index]
            compute_rt[cx][cy] = [num_stat_blocks, is_root, inv_w_bits, eps_bits]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "oc_reader.cpp"),
            core_ranges=kernel_cores_crs,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "oc_writer.cpp"),
            core_ranges=kernel_cores_crs,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "oc_compute.cpp"),
            core_ranges=kernel_cores_crs,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=compute_kernel_config,
        ),
    ]

    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_READY, core_ranges=cb_cores_crs, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_CONSUMED, core_ranges=cb_cores_crs, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_GATHER_PROGRESS, core_ranges=cb_cores_crs, initial_value=0),
    ]

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


def alloc_output(input_tensor):
    """Allocated by the CALLER so a variant that dies in program creation (the L1
    wall is a real outcome here) cannot leak a resident shard into the next case."""
    return ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        input_tensor.memory_config(),
    )


def run(input_tensor, output_tensor, *, stat_rows, apply_rows, pipeline, epsilon=1e-6, compute_kernel_config=None):
    pd = create_program_descriptor(
        input_tensor,
        output_tensor,
        stat_rows=stat_rows,
        apply_rows=apply_rows,
        pipeline=pipeline,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
    )
    return ttnn.generic_op([input_tensor, output_tensor], pd)
