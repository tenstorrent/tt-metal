# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-chip scale measurement for the neighborhood attention op.

Not a pass/fail test -- it prints a table. The question it answers is the one the plan says
must be settled before any performance work: is this kernel compute-bound or memory-bound?

For a breakdown *inside* one call (QK vs softmax vs PV vs reader), see
``test_neighborhood_sdpa_components.py``.

The prediction, from the plan's arithmetic: at stage-5 dimensions a naive per-query-group
DRAM gather moves ~1.4 MB per work item, which should put us DRAM-bound by roughly an order
of magnitude. If that shows up here, L1 residency (plan step 4) is the right next move and
FLOP-level tuning is not. If it does NOT show up, the plan is wrong and step 4 should change.
"""

import time

import pytest
import torch

import ttnn
from models.tt_dit.layers.neighborhood_attention import _query_chunk_bricks

SITES_PER_BRICK = 32

# LTX-2.5 DiffVAE stage 5, 1080p, 6 seconds: 145 frames of 272x480 patches = 18.9M sites.
# Q+K+V+O at 256 channels bf16 is ~39 GB, and a width-sharded slice (145, 272, 80) is still
# ~6.5 GB -- both over one Blackhole's 4.28 GB. So a single chip can only hold a BAND of
# frames, which is why the reference implementation bands over time at all.
#
# Measure bands of the real spatial shape and report per-site, so the result extrapolates to
# the full volume regardless of how width ends up sharded.
STAGE5_FULL_VOLUME = (145, 272, 480)
STAGE5_CONTEXT_WINDOW = (11, 11, 11)
STAGE5_HEAD_COUNT = 4
STAGE5_HEAD_DIM = 64

# (label, volume) -- each sized to fit one chip's DRAM with room to work.
MEASURED_BANDS = [
    ("width_sharded_band", (24, 272, 80)),  # a T-band of one device's width shard
    ("full_width_band", (12, 272, 480)),  # a T-band of the unsharded width
]


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=["mesh_device"])
@pytest.mark.parametrize("stride", [(1, 1, 1), (2, 4, 4)], ids=["stride_one", "stride_equals_brick"])
@pytest.mark.parametrize("band", MEASURED_BANDS, ids=[label for label, _ in MEASURED_BANDS])
def test_report_scale_timing(mesh_device, stride, band):
    band_label, volume = band
    context_window = STAGE5_CONTEXT_WINDOW
    brick = tuple(ttnn.transformer.neighborhood_choose_brick(context_window))
    # The chunk is derived from the stride, exactly as the executor derives it.
    query_chunk_bricks = _query_chunk_bricks(stride, brick)
    plan = ttnn.transformer.neighborhood_plan(
        volume, context_window, stride, brick, query_chunk_bricks=query_chunk_bricks
    )

    brick_count = plan["brick_count"]
    gather_brick_count = plan["gather_brick_count"]
    bricked_site_count = brick_count * SITES_PER_BRICK

    # A chunk's score tiles are live in DST, which holds 8. Prefer one that also divides the
    # gather evenly, so the last chunk is not half padding.
    DST_CAPACITY_TILES = 8
    tiles_per_kv_chunk = next(
        (candidate for candidate in range(DST_CAPACITY_TILES, 0, -1) if gather_brick_count % candidate == 0), 1
    )

    # Site-major: sites are the tile ROW axis and heads are columns, so one tile row is one brick.
    # Timing does not depend on the values, and randn over 163M elements costs ~a minute of
    # host RNG per tensor. One small random block, tiled, keeps the data non-degenerate.
    seed_block = torch.randn(1, 1, SITES_PER_BRICK, STAGE5_HEAD_COUNT * STAGE5_HEAD_DIM)
    filler = seed_block.repeat(1, 1, bricked_site_count // SITES_PER_BRICK, 1)
    tensors = [
        ttnn.from_torch(filler, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device) for _ in range(3)
    ]
    origin_table = torch.tensor(plan["gather_origin_table"], dtype=torch.uint32).reshape(
        1, 1, plan["chunk_count"], plan["gather_origin_columns"]
    )
    origin_on_device = ttnn.from_torch(
        origin_table, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
    )

    def run_once():
        return ttnn.transformer.neighborhood_scaled_dot_product_attention(
            *tensors,
            origin_on_device,
            volume=volume,
            context_window=context_window,
            stride=stride,
            brick=brick,
            query_chunk_bricks=query_chunk_bricks,
            head_count=STAGE5_HEAD_COUNT,
            scale=1.0,
            tiles_per_kv_chunk=tiles_per_kv_chunk,
        )

    run_once()  # warm the program cache and the JIT
    ttnn.synchronize_device(mesh_device)

    iterations = 3
    start = time.perf_counter()
    for _ in range(iterations):
        run_once()
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = (time.perf_counter() - start) * 1000 / iterations

    # What the hardware was asked to do. Scores are counted over the WHOLE gathered span,
    # because that is what the matmul actually evaluates -- masked-out entries included.
    scores = bricked_site_count * STAGE5_HEAD_COUNT * gather_brick_count * SITES_PER_BRICK
    teraflop = scores * STAGE5_HEAD_DIM * 2 * 2 / 1e12  # QK^T and PV, 2 flop per MAC
    exact_scores = bricked_site_count * STAGE5_HEAD_COUNT * (context_window[0] * context_window[1] * context_window[2])

    # Every work item re-reads its whole context window from DRAM: no residency yet.
    kv_gigabytes = (
        bricked_site_count * STAGE5_HEAD_COUNT * gather_brick_count * SITES_PER_BRICK * STAGE5_HEAD_DIM * 2 * 2
    ) / 1e9

    full_sites = STAGE5_FULL_VOLUME[0] * STAGE5_FULL_VOLUME[1] * STAGE5_FULL_VOLUME[2]
    real_site_count = volume[0] * volume[1] * volume[2]
    ms_per_million_sites = elapsed_ms / (real_site_count / 1e6)

    print("")
    print(f"===== {band_label}, one Blackhole, stride {stride} =====")
    print(f"  volume {volume}  window {context_window}  brick {brick}")
    print(f"  bricks {brick_count}   gather {gather_brick_count} bricks/tiles   chunk {tiles_per_kv_chunk} tiles")
    print(f"  work items                 {plan['chunk_count'] * STAGE5_HEAD_COUNT}")
    print(f"  waste vs exact window       {scores / exact_scores:.2f}x")
    print("")
    print(f"  measured                   {elapsed_ms:8.1f} ms")
    print(f"  compute                    {teraflop:8.2f} TFLOP  -> {teraflop / (elapsed_ms / 1000):.1f} TFLOP/s")
    print(f"  KV read (no residency)     {kv_gigabytes:8.1f} GB     -> {kv_gigabytes / (elapsed_ms / 1000):.1f} GB/s")
    print("")
    print(f"  per million sites          {ms_per_million_sites:8.1f} ms")
    print(f"  -> whole stage 5 {STAGE5_FULL_VOLUME}, 8 blocks, 32 chips:")
    print(f"       {ms_per_million_sites * (full_sites / 1e6) * 8 / 32 / 1000:8.2f} s")
    print("")
    print("  Blackhole is roughly ~100 TFLOP/s bf16. Whichever column is near its ceiling is")
    print("  the bottleneck; that decides step 4.")
    print("=" * 62)
