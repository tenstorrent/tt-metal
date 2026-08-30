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

## QuietBox 14 KiB packet A/B sweep

The QuietBox sweep covers every valid-unit remainder modulo its 20 compute lanes by varying from 40 through 59
valid KC units per physical shard, then adds the production 512K endpoint. Both sides used seven measured warm
trace replays; negative delta is faster. The raw replay lists are retained in CI runs
[`33307877873`](https://github.com/tenstorrent/tt-metal/actions/runs/33307877873) (unrotated) and
[`33307877820`](https://github.com/tenstorrent/tt-metal/actions/runs/33307877820) (bounded stride).

| KV prefix | Unrotated ms | Bounded-stride ms | Latency delta |
| ---: | ---: | ---: | ---: |
| 51,200 | 0.204893 | 0.207376 | +1.21% |
| 52,480 | 0.285881 | 0.246399 | -13.81% |
| 53,760 | 0.285364 | 0.249405 | -12.60% |
| 55,040 | 0.285742 | 0.251290 | -12.06% |
| 56,320 | 0.285418 | 0.253361 | -11.23% |
| 57,600 | 0.285844 | 0.274982 | -3.80% |
| 58,880 | 0.285690 | 0.275804 | -3.46% |
| 60,160 | 0.285624 | 0.278201 | -2.60% |
| 61,440 | 0.285696 | 0.279056 | -2.32% |
| 62,720 | 0.285930 | 0.285016 | -0.32% |
| 64,000 | 0.287196 | 0.284970 | -0.78% |
| 65,280 | 0.287073 | 0.284979 | -0.73% |
| 66,560 | 0.288187 | 0.285149 | -1.05% |
| 67,840 | 0.288537 | 0.287293 | -0.43% |
| 69,120 | 0.288531 | 0.288549 | +0.01% |
| 70,400 | 0.291870 | 0.292250 | +0.13% |
| 71,680 | 0.293828 | 0.293302 | -0.18% |
| 72,960 | 0.295457 | 0.296795 | +0.45% |
| 74,240 | 0.300141 | 0.299896 | -0.08% |
| 75,520 | 0.300772 | 0.301800 | +0.34% |
| 524,288 | 1.972519 | 1.973370 | +0.04% |

The production 55K point exceeds both measured goals: latency falls by 11.23% (required at least 9.1%), while
FPU utilization rises from 46.32% to 52.18%, a 12.65% relative gain (required at least 10%). A separate
three-replay CI session measured 52.26%. The checked-in 55K target is therefore raised from 46.26% to 52.22%,
the midpoint of the two post-change session medians, with the existing symmetric +/-2% margin. No other target
changes: 512K is neutral and every sweep point stays within the 2% no-regression bound.

## QuietBox default-packet A/B guard

This guard omitted `fabric_router_config` and used seven warm trace replays per point. The raw replay lists are
retained in CI runs [`33308688927`](https://github.com/tenstorrent/tt-metal/actions/runs/33308688927)
(unrotated) and [`33309333874`](https://github.com/tenstorrent/tt-metal/actions/runs/33309333874)
(bounded stride). The measurement-only jobs intentionally used the 14 KiB target table, so their assertions are
not meaningful for the default packet configuration; the recorded durations are the A/B acceptance data.

| KV prefix | Unrotated ms | Bounded-stride ms | Latency delta |
| ---: | ---: | ---: | ---: |
| 56,320 | 0.297491 | 0.275300 | -7.46% |
| 524,288 | 2.204549 | 2.202112 | -0.11% |

Both endpoints remain inside the 2% no-regression bound; the underfilled 55K case improves substantially.

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

## Analytical sweep audit

The `PrefixSweepRecordsUnitTileAndArrivalWaveBalance` host test derives these values with the production
`ring_writes_for`, `RingIdSequencer`, and block-cyclic logical-tile mapping. `Units/rem` is the valid KC-unit
count per physical shard and its remainder modulo the physical lane count. Ratios are the worst rank's maximum
lane load divided by mean lane load; `old -> new` compares unrotated and bounded-stride schedules.

<details>
<summary>QuietBox: every 20-lane remainder</summary>

| KV prefix | Units/rem | Nonempty KC max/mean | Valid-tile max/mean |
| ---: | ---: | ---: | ---: |
| 51,200 | 40/0 | 1.000 -> 1.000 | 1.000 -> 1.000 |
| 52,480 | 41/1 | 1.463 -> 1.220 | 1.463 -> 1.220 |
| 53,760 | 42/2 | 1.429 -> 1.190 | 1.429 -> 1.190 |
| 55,040 | 43/3 | 1.395 -> 1.163 | 1.395 -> 1.163 |
| 56,320 | 44/4 | 1.364 -> 1.136 | 1.364 -> 1.136 |
| 57,600 | 45/5 | 1.333 -> 1.222 | 1.333 -> 1.222 |
| 58,880 | 46/6 | 1.304 -> 1.196 | 1.304 -> 1.196 |
| 60,160 | 47/7 | 1.277 -> 1.170 | 1.277 -> 1.170 |
| 61,440 | 48/8 | 1.250 -> 1.146 | 1.250 -> 1.146 |
| 62,720 | 49/9 | 1.224 -> 1.224 | 1.224 -> 1.224 |
| 64,000 | 50/10 | 1.200 -> 1.200 | 1.200 -> 1.200 |
| 65,280 | 51/11 | 1.176 -> 1.176 | 1.176 -> 1.176 |
| 66,560 | 52/12 | 1.154 -> 1.154 | 1.154 -> 1.154 |
| 67,840 | 53/13 | 1.132 -> 1.132 | 1.132 -> 1.132 |
| 69,120 | 54/14 | 1.111 -> 1.111 | 1.111 -> 1.111 |
| 70,400 | 55/15 | 1.091 -> 1.091 | 1.091 -> 1.091 |
| 71,680 | 56/16 | 1.071 -> 1.071 | 1.071 -> 1.071 |
| 72,960 | 57/17 | 1.053 -> 1.053 | 1.053 -> 1.053 |
| 74,240 | 58/18 | 1.034 -> 1.034 | 1.034 -> 1.034 |
| 75,520 | 59/19 | 1.017 -> 1.017 | 1.017 -> 1.017 |
| 524,288 | 410/10 | 1.024 -> 1.024 | 1.025 -> 1.025 |

</details>

<details>
<summary>LoudBox: every 22-lane remainder</summary>

| KV prefix | Units/rem | Nonempty KC max/mean | Valid-tile max/mean |
| ---: | ---: | ---: | ---: |
| 56,320 | 22/0 | 1.000 -> 1.000 | 1.000 -> 1.000 |
| 58,880 | 23/1 | 1.913 -> 1.196 | 1.913 -> 1.196 |
| 61,440 | 24/2 | 1.833 -> 1.146 | 1.833 -> 1.146 |
| 64,000 | 25/3 | 1.760 -> 1.320 | 1.760 -> 1.320 |
| 66,560 | 26/4 | 1.692 -> 1.269 | 1.692 -> 1.269 |
| 69,120 | 27/5 | 1.630 -> 1.426 | 1.630 -> 1.426 |
| 71,680 | 28/6 | 1.571 -> 1.375 | 1.571 -> 1.375 |
| 74,240 | 29/7 | 1.517 -> 1.422 | 1.517 -> 1.422 |
| 76,800 | 30/8 | 1.467 -> 1.375 | 1.467 -> 1.375 |
| 79,360 | 31/9 | 1.419 -> 1.419 | 1.419 -> 1.419 |
| 81,920 | 32/10 | 1.375 -> 1.375 | 1.375 -> 1.375 |
| 84,480 | 33/11 | 1.333 -> 1.333 | 1.333 -> 1.333 |
| 87,040 | 34/12 | 1.294 -> 1.294 | 1.294 -> 1.294 |
| 89,600 | 35/13 | 1.257 -> 1.257 | 1.257 -> 1.257 |
| 92,160 | 36/14 | 1.222 -> 1.222 | 1.222 -> 1.222 |
| 94,720 | 37/15 | 1.189 -> 1.189 | 1.189 -> 1.189 |
| 97,280 | 38/16 | 1.158 -> 1.158 | 1.158 -> 1.158 |
| 99,840 | 39/17 | 1.128 -> 1.128 | 1.128 -> 1.128 |
| 102,400 | 40/18 | 1.100 -> 1.100 | 1.100 -> 1.100 |
| 104,960 | 41/19 | 1.073 -> 1.073 | 1.073 -> 1.073 |
| 107,520 | 42/20 | 1.048 -> 1.048 | 1.048 -> 1.048 |
| 110,080 | 43/21 | 1.023 -> 1.023 | 1.023 -> 1.023 |
| 524,288 | 205/7 | 1.073 -> 1.060 | 1.074 -> 1.061 |

</details>

For every sweep point, the same test also records the worst-rank per-lane work tail beginning with each arrival
wave, separately in nonempty KC units and valid K tiles. The vectors below show the production endpoints and
the LB diagnostic; entries run from the local wave through the final opposite-shard wave.

| Case | Nonempty KC tail, old -> new | Valid-tile tail, old -> new |
| :--- | :--- | :--- |
| QB 55K | `1.364,1.364,1.364` -> `1.136,1.212,1.364` | `1.364,1.364,1.364` -> `1.136,1.212,1.364` |
| QB 512K | `1.024,1.024,1.024` -> `1.024,1.024,1.024` | `1.025,1.026,1.026` -> `1.025,1.026,1.026` |
| LB 55K | `1.000,1.000,1.000,1.000,1.000` -> `1.000,1.000,1.000,1.000,1.000` | `1.000,1.000,1.000,1.000,1.000` -> `1.000,1.000,1.000,1.000,1.000` |
| LB 58,880 | `1.913,1.913,1.913,1.913,1.913` -> `1.196,1.230,1.339,1.594,1.913` | `1.913,1.913,1.913,1.913,1.913` -> `1.196,1.230,1.339,1.594,1.913` |
| LB 512K | `1.073,1.073,1.073,1.073,1.073` -> `1.060,1.073,1.073,1.073,1.073` | `1.074,1.074,1.074,1.074,1.074` -> `1.061,1.067,1.074,1.074,1.074` |

The final single-shard tail cannot be redistributed across arrival waves, so its ratio is intentionally unchanged.

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
- `RingIndexerScoreSchedule.*`: 6/6 pass. The suite emits the complete analytical sweep and Ring-32 audit as
  gtest properties in addition to enforcing coverage, order, partial-final-KC clamping, nonempty small-capacity
  lanes, and no-worse maximum nonempty-KC load.
- Fused Ring Indexer device suite: 14/14 pass on the 8-device Blackhole box, including Ring-8 partial readiness,
  changing-`kv_len` program-cache reuse, block-cyclic K, straddle, BF16 K, and BFP8 K.
- Final Blackhole multi-card CI run
  [`33308731203`](https://github.com/tenstorrent/tt-metal/actions/runs/33308731203) passed the centered 55K and
  unchanged 512K performance bands. Its complete `tests/nightly/blackhole/sdpa/` selection passed 125 tests and
  skipped 99, including every CI-scheduled case in `test_ring_joint_sdpa.py`.
- Wormhole T3K run [`33307361496`](https://github.com/tenstorrent/tt-metal/actions/runs/33307361496) passed all
  selected TTNN, multiprocess, TT-Metal multiprocess, and UDM legs.
- The checked-in perf test changes only the justified QuietBox 55K target. All sweep/default-packet harness
  modifications were measurement-only and are absent from the implementation branch.

## Ring-32 analytical handoff

The host test constructs the exact production arrival order for every local rank with `ring_writes_for` and
`RingIdSequencer`. For both 20 and 22 lanes it checks every prefix from 1 through 32,640 K tiles, using the exact
5-tile block-cyclic logical mapping.

| Lanes | Ranks | Prefixes/rank | Coverage/order | Worst new/old max KC | Worst KC max/mean (prefix, rank) | Worst tile max/mean (prefix, rank) | Full-capacity max KC, old -> new |
| ---: | ---: | ---: | :---: | ---: | :--- | :--- | ---: |
| 20 | 32 | 32,640 | pass | 1.000 | 20.000 (1 tile, rank 0) | 20.000 (1 tile, rank 0) | 192 -> 164 |
| 22 | 32 | 32,640 | pass | 1.000 | 22.000 (1 tile, rank 0) | 22.000 (1 tile, rank 0) | 160 -> 152 |

The 1.000 new/old maximum ratio and worst maximum/mean values occur at the one-tile prefix, where both schedules
necessarily place the only valid tile on one lane. The valid-tile distribution has the same 1.000 worst
new/old maximum ratio, also at prefix 1 and rank 0, for both geometries. No prefix has a worse maximum nonempty
KC load than the unrotated schedule. This is analytical validation only; a future
32-device run must still cover nonzero-K correctness, dynamic-prefix cache reuse, the remainder sweep, 55K and
512K performance, both packet configurations, and progress through all 17 arrival waves.

## Final review gate

The hardware, performance, no-regression, and analytical acceptance gates are complete. Before merge, fresh
independent Codex Sol and Claude Fable reviews must explicitly approve both correctness/goal attainment and
cleanup/simplification/refactoring quality on the same final revision.
