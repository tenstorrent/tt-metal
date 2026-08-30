# Ring Indexer Wave-Balanced Scheduling Results

## Revision under test

- Baseline: `61f9ff45a617f9640664d571cad3cda2e3def8c2`
- Implementation: row-block-preserving, bounded-stride arrival-wave rotation
- Local hardware: 8-device Blackhole LoudBox, physical `12 x 10` worker grid and 22 block-column compute lanes
- Execution: warm trace replay; medians below use seven measured replays per point
- Fabric: Ring topology with two links per direction

The implementation assigns each arrival wave a cyclic compute-column offset while preserving its row block.
It does not inspect runtime `kv_len`, add a program variant, or change a kernel ABI. Linear and Ring-2 retain
the prior work assignment.

## LoudBox 14 KiB packet A/B sweep

The sweep covers every valid-unit remainder modulo 22 by varying from 22 through 43 valid KC units per shard,
then adds the production 512K endpoint. Negative delta is faster.

| KV prefix | Unrotated ms | Bounded-stride ms | Latency delta |
| ---: | ---: | ---: | ---: |
| 56,320 | 0.204 | 0.205 | +0.5% |
| 58,880 | 0.373 | 0.242 | -35.1% |
| 61,440 | 0.372 | 0.242 | -34.9% |
| 64,000 | 0.373 | 0.285 | -23.6% |
| 66,560 | 0.372 | 0.286 | -23.1% |
| 69,120 | 0.373 | 0.332 | -11.0% |
| 71,680 | 0.372 | 0.334 | -10.2% |
| 74,240 | 0.373 | 0.357 | -4.3% |
| 76,800 | 0.372 | 0.361 | -3.0% |
| 79,360 | 0.373 | 0.371 | -0.5% |
| 81,920 | 0.373 | 0.371 | -0.5% |
| 84,480 | 0.373 | 0.371 | -0.5% |
| 87,040 | 0.374 | 0.371 | -0.8% |
| 89,600 | 0.374 | 0.371 | -0.8% |
| 92,160 | 0.374 | 0.371 | -0.8% |
| 94,720 | 0.375 | 0.372 | -0.8% |
| 97,280 | 0.375 | 0.373 | -0.5% |
| 99,840 | 0.377 | 0.375 | -0.5% |
| 102,400 | 0.379 | 0.377 | -0.5% |
| 104,960 | 0.380 | 0.379 | -0.3% |
| 107,520 | 0.388 | 0.383 | -1.3% |
| 110,080 | 0.388 | 0.387 | -0.3% |
| 524,288 | 1.867 | 1.883 | +0.9% |

The plan's 58,880 diagnostic improves by 35.1%, exceeding the required 10%. Every sweep point remains inside
the 2% no-regression bound. The production 55K and 512K endpoints remain inside their checked-in symmetric 2%
FPU-utilization bands, so no LoudBox target adjustment is justified.

## LoudBox default-packet A/B guard

This run omitted `fabric_router_config`; it therefore used the runtime's default packet configuration rather
than the checked-in test's explicit 14 KiB setting.

| KV prefix | Unrotated ms | Bounded-stride ms | Latency delta |
| ---: | ---: | ---: | ---: |
| 56,320 | 0.218 | 0.216 | -0.9% |
| 524,288 | 1.943 | 1.941 | -0.1% |

Both endpoints remain inside the 2% no-regression bound.

## Rejected wider rotation

An earlier evenly-spaced column rotation produced the same 58,880 analytical balance but regressed the 14 KiB
512K median from 1.867 ms to 1.929 ms (+3.3%). A still earlier full-lane rotation also changed row-block parity
and measured about 58.0% rather than the ~60.0% baseline FPU utilization at 512K. Both were rejected. The final
bounded stride minimizes displacement and preserves row-block DRAM pairing.

## Correctness and cache validation

- `ttnn` and `unit_tests_ttnn` build successfully with warnings as errors.
- `RingIndexerScoreSchedule.*`: 4/4 pass.
- Fused Ring Indexer device suite: 14/14 pass on the 8-device Blackhole box, including Ring-8 partial readiness,
  changing-`kv_len` program-cache reuse, block-cyclic K, straddle, BF16 K, and BFP8 K.
- The existing perf test file is unchanged; all sweep/default-packet modifications were measurement-only and
  reverted after collection.

## Ring-32 analytical handoff

The host test constructs the exact production arrival order for every local rank with `ring_writes_for` and
`RingIdSequencer`. For both 20 and 22 lanes it checks every prefix from 1 through 32,640 K tiles, using the exact
5-tile block-cyclic logical mapping.

| Lanes | Ranks | Prefixes per rank | Coverage/order | Worst rotated/current max-unit ratio | Full-capacity max units, old -> new |
| ---: | ---: | ---: | :---: | ---: | ---: |
| 20 | 32 | 32,640 | pass | 1.000 | 192 -> 164 |
| 22 | 32 | 32,640 | pass | 1.000 | 160 -> 152 |

The 1.000 worst ratio occurs at tiny prefixes where both schedules necessarily place the only nonempty unit on
one lane. No prefix is worse than the unrotated schedule. This is analytical validation only; a future
32-device run must still cover nonzero-K correctness, dynamic-prefix cache reuse, the remainder sweep, 55K and
512K performance, both packet configurations, and progress through all 17 arrival waves.

## Pending acceptance

- QuietBox Ring-4 hardware measurements and the 20-lane remainder sweep.
- Blackhole and Wormhole CI selections required by the plan, including every scheduled case in
  `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py`.
- Fresh independent Codex Sol and Claude Fable reviews of the final code and results, with explicit approval of
  correctness/goal attainment and cleanup/simplification/refactoring quality.
