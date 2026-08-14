# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""overlap_combine — the bake-off harness.  Correctness is the ONLY pass/fail.

Run (device kernel ns come from the profiler CSV the wrapper prints):

    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/overlap_combine/test_overlap_combine.py

Each case runs ONCE — device kernel time has no warm-up transient.

VARIANTS (all at the SAME fixed precision contract: bf16 x, fp32 stat tiles,
HiFi2, fp32_dest_acc_en=False):

    base_bN     the op's CURRENT structure — serial loop, stat block == apply
                block == N tile-rows, one combine round trip per block.
    pipe_bN     lever 1: software-pipelined block loop (stat(b+1) issued before
                waiting for 1/rms(b)), depth-2 stat/gather/bcast/recip CBs.
    coarseSB_bN lever 2: ONE combine round trip per SB tile-rows while the apply
                pass still walks N at a time.
    pipe_coarse lever 1 + lever 2 together (usually L1-blocked: the landing
                buffer is s * SB fp32 tiles and doubling it is 1 MB at SB=16).

The correctness gate is the thing a mis-pipelined loop breaks SILENTLY: every
block must get ITS OWN 1/rms.  A schedule that lets block b scale with block
b-1's (or b+1's) statistic still runs, still hangs nothing, and just produces
subtly wrong numbers — so PCC vs torch is checked on every variant.

=============================================================================
MEASURED — Blackhole p150b @1350 MHz, DEVICE KERNEL DURATION [ns], two
independent fresh runs (run2 / run1).  Every cell at the same fixed precision
contract; PCC as noted.  "L1 KB" is per core: resident shards + all CBs.
=============================================================================

focus  (1,1,8192,1024) BLOCK [1024,128] (8,8)  s=8 S=4  32 tile-rows/core, 64 cores
  variant              round trips     ns          vs base_b8   CB KB  L1 KB   PCC
  base_b4                    8      72302 / 72312     +6.2%      178    690   .999844
  pipe_b4                    8      59640 / 59605    -12.4%      354    866   .999844
  base_b8  (BASELINE)        4      68091 / 68118        —       354    866   .999844
  pipe_b8  (RECOMMENDED)     4      57935 / 57923    -14.9%      706   1218   .999844
  base_b16                   2      66026 / 66059     -3.0%      704   1216   .999844
  coarse16_b8                2      66043 / 66035     -3.0%      704   1216   .999844
  pipe_b16                   2         L1 WALL          —       1410   1922     —
  coarse32_b8 / base_b32     1         L1 WALL          —       1410   1922     —
  pipe_coarse16_b8           2         L1 WALL          —       1410   1922     —

s4     (1,1,8192, 512) BLOCK [1024,128] (4,8)  s=4 S=4  32 tile-rows/core, 32 cores
  base_b8  (BASELINE)        4      59642 / 59642        —       226    738   1.000000
  pipe_b8                    4      54058 / 54106     -9.4%      450    962   1.000000
  coarse16_b8                2      57879 / 57870     -3.0%      450    962   1.000000
  coarse32_b8                1      57013 / 57031     -4.4%      898   1410   1.000000
  pipe_coarse16_b8           2      54361 / 54336     -8.9%      898   1410   1.000000

short  (1,1,2048,1024) BLOCK [ 256,128] (8,8)  s=8 S=4   8 tile-rows/core, 64 cores
  base_b2                    4      21165             +19.0%      90    218   1.000000
  pipe_b2                    4      17971              +1.1%     178    306   1.000000
  base_b4  (BASELINE)        2      18887 (ref)           —      176    304   1.000000
  pipe_b4  (BEST)            2      16996             -10.0%     350    478   1.000000
  base_b8                    1      17781              -5.9%     350    478   1.000000
  coarse8_b4                 1      17916              -5.1%     350    478   1.000000
  pipe_coarse8_b4            1      17843    (== coarse8_b4: 1 round trip, so the
                                              program is byte-identical — FLAT)

decode (1,1,  32,1024) WIDTH [  32,128] (8,1)  s=8 S=4   1 tile-row/core,  8 cores
  base_b1                    1       3993 / 3958          —        46     62   .999995
  pipe_b1                    1       4000 / 4063        flat       46     62   .999995
    ^ ONE round trip => nothing to pipeline, so the host keeps depth 1 and the two
      programs are BYTE-IDENTICAL.  The 0.2% / 2.7% spread IS the noise band here.

-----------------------------------------------------------------------------
READING
-----------------------------------------------------------------------------
* BASELINE FIDELITY.  The base_b4/b8/b16 column reproduces the op's own
  block_rows sweep almost exactly (op: 76188 / 72041 / 70039 with gamma; deltas
  -4147 and -2002 vs this bench's -4211 and -2065).  Absolute ns sit ~4 us below
  the op's because gamma is held out.
* LEVER 1 (pipeline) is the win, and it is worth ~the whole GATHER INCAST.  The
  focus win of 10156 ns is 85% of the 11950 ns the op's own ablation attributes
  to the incast: stat(b+1) leaves while the root reduces block b, so by the time
  the root broadcasts 1/rms(b) the block-b+1 gather has already landed.  It
  scales with s (s=8 -15%, s=4 -9%) and with the number of round trips available
  to overlap (8 rt -12%, 4 rt -15%, 2 rt -10%, 1 rt flat), which is what the
  mechanism predicts.
* LEVER 2 (coarse stat) is a small, real win (-3% .. -4.4%) and it is ENTIRELY
  the round-trip count.  Decoupling the stat block from the apply block buys
  NOTHING on a resident shard: coarse16_b8 == base_b16 (66043 vs 66026) in time
  AND in L1, because the apply block's L1 cost is zero here (cb_output IS the
  shard) while the coarse stat's cost is the s x SB landing buffer, which
  decoupling does not avoid.  Decoupling would only pay on the INTERLEAVED path,
  where cb_input/cb_output are real block-sized CBs.
* THE TWO LEVERS ARE SUBSTITUTES, NOT ADDITIVE.  Where L1 lets both run (s4),
  pipe+coarse16 (54361) is no better than pipe alone (54058).  Both remove
  exposed combine round trips; there is an interior optimum at ~2-4 round trips
  (short: 4 rt 17971, 2 rt 16996, 1 rt 17781 — a single coarse round trip LOSES
  to two pipelined ones, because one round trip has nothing left to overlap).
* THE L1 WALL, verbatim: "Statically allocated circular buffers in program N
  clash with L1 buffers on core range [0-0 - 7-7]. L1 buffer allocated at
  1048576 and static circular buffer region ends at 1555328".  The two resident
  shards take the top 512 KB of L1, so the CB region is ~937 KB.  depth-2 gather
  at SB=16 (or depth-1 at SB=32) needs 1410 KB and is INEXPRESSIBLE.  The gather
  landing (s x SB fp32 tiles) is the whole term — cutting the stat payload is
  what would unlock the pipeline's better operating point (2 round trips).
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from eval.sharding import shard_config
from ttnn.operations.rms_norm.perf_experiments.overlap_combine import bench

_ML = ttnn.TensorMemoryLayout

# The perf group's fixed config.  NOT a lever.
TARGET_FIDELITY = ttnn.MathFidelity.HiFi2
TARGET_FP32_ACC = False
PCC_THRESHOLD = 0.9995  # the focus case's soft gate
EPS = 1e-6


def target_compute_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=TARGET_FIDELITY,
        fp32_dest_acc_en=TARGET_FP32_ACC,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# Geometries.  (shape, memory_layout, shard_shape, core_grid) -> shard_rows, s, S
#
#   focus     (1,1,8192,1024) BLOCK [1024,128] (8,8): 32 shard tile-rows, s=8, S=4
#             — the perf-flagged case, 64 cores, 8 row-groups of 8.
#   s4        (1,1,8192, 512) BLOCK [1024,128] (4,8): 32 shard tile-rows, s=4, S=4
#             — the same row-group depth with HALF the combine fan-in.
#   decode    (1,1,  32,1024) WIDTH [  32,128] (8,1):  1 shard tile-row,  s=8, S=4
#             — ONE round trip, nothing to pipeline.  Must be flat, not worse.
# ---------------------------------------------------------------------------
#   short     (1,1,2048,1024) BLOCK [ 256,128] (8,8):  8 shard tile-rows, s=8, S=4
#             — a SHALLOW row-group: 4x less local work per core to hide the
#             combine behind, and small enough shards that every stat_rows fits
#             L1 (so the levers can be measured to their limit).
GEOMETRIES = {
    "focus": ((1, 1, 8192, 1024), _ML.BLOCK_SHARDED, [1024, 128], (8, 8)),
    "s4": ((1, 1, 8192, 512), _ML.BLOCK_SHARDED, [1024, 128], (4, 8)),
    "short": ((1, 1, 2048, 1024), _ML.BLOCK_SHARDED, [256, 128], (8, 8)),
    "decode": ((1, 1, 32, 1024), _ML.WIDTH_SHARDED, [32, 128], (8, 1)),
}

# (geometry, stat_rows, apply_rows, pipeline)
CASES = [
    # ---- the focus shape: the menu ----
    pytest.param("focus", 8, 8, 0, id="focus_base_b8"),  # THE BASELINE (op's shipped block_rows)
    pytest.param("focus", 8, 8, 1, id="focus_pipe_b8"),  # lever 1
    pytest.param("focus", 16, 8, 0, id="focus_coarse16_b8"),  # lever 2
    pytest.param("focus", 32, 8, 0, id="focus_coarse32_b8"),  # lever 2, max (expect L1 wall)
    pytest.param("focus", 16, 8, 1, id="focus_pipe_coarse16_b8"),  # levers 1+2 (expect L1 wall)
    # ---- num_blocks sweep at the focus shape (block_rows 4 / 16 / 32) ----
    pytest.param("focus", 4, 4, 0, id="focus_base_b4"),
    pytest.param("focus", 4, 4, 1, id="focus_pipe_b4"),
    pytest.param("focus", 16, 16, 0, id="focus_base_b16"),
    pytest.param("focus", 16, 16, 1, id="focus_pipe_b16"),
    pytest.param("focus", 32, 32, 0, id="focus_base_b32"),  # num_blocks == 1
    pytest.param("focus", 32, 32, 1, id="focus_pipe_b32"),  # nothing to pipeline
    # ---- s = 4 ----
    pytest.param("s4", 8, 8, 0, id="s4_base_b8"),
    pytest.param("s4", 8, 8, 1, id="s4_pipe_b8"),
    pytest.param("s4", 16, 8, 0, id="s4_coarse16_b8"),
    pytest.param("s4", 32, 8, 0, id="s4_coarse32_b8"),
    pytest.param("s4", 16, 8, 1, id="s4_pipe_coarse16_b8"),
    # ---- a SHALLOW row-group (8 tile-rows/core): less local work to hide ----
    pytest.param("short", 2, 2, 0, id="short_base_b2"),
    pytest.param("short", 2, 2, 1, id="short_pipe_b2"),
    pytest.param("short", 4, 4, 0, id="short_base_b4"),
    pytest.param("short", 4, 4, 1, id="short_pipe_b4"),
    pytest.param("short", 8, 4, 0, id="short_coarse8_b4"),
    pytest.param("short", 8, 4, 1, id="short_pipe_coarse8_b4"),  # 1 round trip => flat
    pytest.param("short", 8, 8, 0, id="short_base_b8"),
    # ---- the decode regime: one round trip, must be FLAT ----
    pytest.param("decode", 1, 1, 0, id="decode_base_b1"),
    pytest.param("decode", 1, 1, 1, id="decode_pipe_b1"),
]


@pytest.mark.parametrize("geom,stat_rows,apply_rows,pipeline", CASES)
def test_overlap_combine(device, geom, stat_rows, apply_rows, pipeline):
    shape, memory_layout, shard_shape, core_grid = GEOMETRIES[geom]

    torch.manual_seed(42)
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)

    memory_config = shard_config(
        shard_shape,
        core_grid,
        memory_layout,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    x = ttnn.from_torch(
        torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )

    p = bench.plan(x)
    l1 = bench.l1_report(x, x, stat_rows=stat_rows, pipeline=pipeline)
    num_stat_blocks = p["shard_rows"] // stat_rows
    print(
        f"\n[overlap_combine] {geom} s={p['num_slices']} S={p['slice_tiles']} "
        f"shard_rows={p['shard_rows']} stat_rows={stat_rows} apply_rows={apply_rows} "
        f"pipeline={pipeline} round_trips={num_stat_blocks} "
        f"L1/core: shards={l1['shard_bytes'] // 1024}KB cbs={l1['cb_bytes'] // 1024}KB "
        f"total={l1['total_bytes'] // 1024}KB"
    )

    # Every case allocates a fresh pair of resident L1 shards, and the device is
    # module-scoped so the bake-off shares one L1 heap.  Without an explicit
    # deallocate the second half of the sweep dies in the allocator (measured) —
    # which looks exactly like an L1 verdict on the variant and is NOT one.
    out_tt = bench.alloc_output(x)
    try:
        out = ttnn.to_torch(
            bench.run(
                x,
                out_tt,
                stat_rows=stat_rows,
                apply_rows=apply_rows,
                pipeline=pipeline,
                epsilon=EPS,
                compute_kernel_config=target_compute_config(),
            )
        ).to(torch.float32)
    finally:
        ttnn.deallocate(out_tt)
        ttnn.deallocate(x)

    xf = torch_x.to(torch.float32)
    expected = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
    a, b = out.flatten(), expected.flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    print(f"[overlap_combine] PCC = {pcc:.6f}")
    assert pcc > PCC_THRESHOLD, f"{geom}/{stat_rows}/{apply_rows}/pipe{pipeline}: PCC {pcc}"
