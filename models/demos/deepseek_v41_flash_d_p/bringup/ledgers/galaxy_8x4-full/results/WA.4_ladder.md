# WA.4: wo_a bank-local reads, the accept measurement (DS41_WOA_BANK off vs on)

## Verdict data (2026-11-27, attempt 1)

- **Accuracy (switch on):** off == on bit-identical over **all 40 layers** (DECODER_SKIP=off, one load, both copies)
  on single 8192, packed 4x2k and single 5120 (every layer output and every served record: 0.0), on-repeats 0.0,
  real160 (40 real wo_a inputs at the 160-row knobs) 0.0 (test_woa_offon, WOA_OFFON_LAYERS=all). The fresh 40-layer
  reference (test_chunk8192_reference) and the fresh-process determinism run in the accept gate.
- **Device win, same load (test_woa_offon timing, layers 0-20, median of 3 profiled runs a side, alternating):** wo_a
  3.058 -> 2.656 ms a single-8192 chunk (-0.40), 3.067 -> 2.662 (4x2k, -0.41), 3.062 -> 2.651 (single 5120 chunk in the
  8192 load, -0.41); whole chunk 85.41 -> 84.82 (-0.60), 94.72 -> 94.20 (-0.52), 98.41 -> 97.73 ms (-0.68).
- **Host a wo_a call** (warm, best of 2): 97.4 / 97.8 us (single 8192), 109.0 / 116.2 (4x2k), 98.8 / 106.7 (5120);
  WA.3's runs spread 104.9-159.4 us, so +7-8 us is within the spread (at most +0.16 ms a chunk on 20 calls).
- **Ladders (on side, this session):** the tables below. The single-8192 attention section is 0.25-0.53 ms lower at
  every rung than the last accepted ladder (PB.MM256); packed device deltas sit within the sessions' drift.
- **Load time:** the bank copy costs +1.97 s over the interleaved copy for all 40 layers (bank 4.37 s, interleaved
  2.40 s, host prep 3.34 s common to both); WA.3's 21-layer number was +1.39 s. Not cached.
- **Weight DRAM a chip, wo_a:** 35.65 MB a layer. Shipped load (DS41_WOA_BANK_COPIES=selected, OV-77): one copy,
  1.33 GiB for 40 layers (0.70 GiB for the 21-layer perf load): the same as today, whichever layout is selected. A/B /
  test load (both): 2.66 GiB for 40 (+1.33 GiB), only while an off / on test holds both (test_woa_offon adds the copy and
  drops it after). The ladders' own `dram_load_gib_max` (perf load, held by the resident runner): 10.75 GiB (8192),
  9.16 GiB (5120).
- **Cold no-rebuild (strict, fresh process, switch on):** PASS on every path, 0 program builds after warm-up.

**Off side of the ladders (re-rendered at WA.5, OV-80):** after F306 the orchestrator's A/B measured both sides,
and the tables below are its own off and on runs (no proxy). But the off side's ladder, 5120 and no-rebuild units ran
with DS41_WOA_BANK=on: each gate command carries its own `DS41_WOA_BANK=on` prefix, and orchestrator.ab_batches lets a
segment's prefix win over the side's env (`<run>/ab/WA.4/old/batch0.json`: every such entry's env has
DS41_WOA_BANK=on; DS41-397). Those tables are on vs on: their deltas (single 8192 attention -0.22..+0.23 ms, device
-0.58..+0.26 ms) are session noise, where a real off vs on would show the attention section ~0.40 ms lower. The real
off vs on of this A/B is test_woa_offon (the switch per call inside one process): wo_a 3.058 -> 2.656 ms a chunk,
chunk 85.41 -> 84.82 ms (single 8192).

Rendered by tests/woa/woa_ladder_report.py from the A/B runs of WA.4 (/scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/runs/galaxy_8x4-full/run1/ab/WA.4; old = off, new = on,
the gate's commands, one resident load a side's chunk size, the switch call-time). Ladder method as PB.LAD
(tests/packed/ladder_common.py): steady = mid-request chunk, resumed = first chunk after a prefix hit; 10 timed
runs, median, program-cache misses refused; device ms = one profiled run (slowest chip); exposed host = wall -
device. Layers 0-20 (DECODER_SKIP=decode). `-` = that side has not run (yet).

**Switch as run (the sides' batch plans):** the off side ran test_no_rebuild with DS41_WOA_BANK=on, test_packed_walls with DS41_WOA_BANK=on, test_skip_positions__BRINGUP_CHUNK-5120 with DS41_WOA_BANK=on, test_skip_positions__BRINGUP_CHUNK-8192__43244588 with DS41_WOA_BANK=on: the gate command's own `DS41_WOA_BANK=on` prefix wins over the side's env (orchestrator.ab_batches, a segment's prefix wins). Those tables are on vs on: their deltas are session noise, not the change. The one real off vs on is test_woa_offon (it switches per call itself).

## Single request, 8192 tokens (test_skip_positions, BRINGUP_CHUNK=8192)

| history | steady wall ms off / on (on - off) | steady tok/s off / on (on - off) | resumed wall ms off / on (on - off) | steady device ms off / on (on - off) | steady exposed host ms off / on (on - off) | attention device ms off / on (on - off) |
|---|---|---|---|---|---|---|
| 51200 | 97.5 / 99.9 (+2.4) | 84020 / 82002 (-2018) | 102.2 / 101.8 (-0.4) | 82.29 / 82.39 (+0.10) | 15.2 / 17.5 (+2.3) | 26.98 / 27.00 (+0.02) |
| 131072 | 99.0 / 98.0 (-1.0) | 82748 / 83592 (+844) | 102.6 / 102.7 (+0.1) | 84.41 / 84.20 (-0.21) | 14.6 / 13.8 (-0.8) | 26.92 / 26.81 (-0.11) |
| 139264 | 98.2 / 98.1 (-0.1) | 83422 / 83507 (+85) | 101.8 / 102.6 (+0.8) | 84.03 / 83.94 (-0.09) | 14.2 / 14.2 (+0.0) | 26.84 / 26.86 (+0.02) |
| 262144 | 101.3 / 100.3 (-1.0) | 80869 / 81675 (+806) | 104.9 / 103.7 (-1.2) | 89.96 / 89.38 (-0.58) | 11.3 / 10.9 (-0.4) | 27.13 / 27.00 (-0.13) |
| 524288 | 104.7 / 103.7 (-1.0) | 78243 / 78997 (+754) | 107.1 / 108.8 (+1.7) | 93.25 / 93.51 (+0.26) | 11.5 / 10.2 (-1.3) | 26.86 / 27.02 (+0.16) |
| 786432 | 114.9 / 114.4 (-0.5) | 71297 / 71608 (+312) | 119.3 / 118.8 (-0.5) | 103.84 / 103.89 (+0.05) | 11.1 / 10.5 (-0.6) | 27.28 / 27.06 (-0.22) |
| 1024000 | 120.1 / 120.3 (+0.2) | 68210 / 68096 (-113) | 120.6 / 124.0 (+3.4) | 107.89 / 107.95 (+0.06) | 12.2 / 12.3 (+0.1) | 27.19 / 27.42 (+0.23) |

## Chip-aligned 4 x 2048 (lanes4x2k, test_packed_walls)

| history | steady wall ms off / on (on - off) | steady tok/s off / on (on - off) | resumed wall ms off / on (on - off) | steady device ms off / on (on - off) | steady exposed host ms off / on (on - off) | attention device ms off / on (on - off) |
|---|---|---|---|---|---|---|
| 51200 | 111.3 / 108.4 (-2.9) | 73603 / 75572 (+1969) | 108.3 / 107.8 (-0.5) | 89.09 / 90.08 (+0.99) | 22.2 / 18.3 (-3.9) | 26.79 / 26.88 (+0.09) |
| 131072 | 111.0 / 110.2 (-0.8) | 73802 / 74338 (+536) | 110.2 / 108.3 (-1.9) | 90.85 / 91.53 (+0.68) | 20.2 / 18.7 (-1.5) | 27.66 / 28.10 (+0.44) |
| 139264 | 111.2 / 109.5 (-1.7) | 73669 / 74813 (+1144) | 113.4 / 110.8 (-2.6) | 91.72 / 92.07 (+0.35) | 19.5 / 17.4 (-2.1) | 27.62 / 27.59 (-0.03) |
| 262144 | 115.2 / 113.3 (-1.9) | 71111 / 72304 (+1192) | 115.3 / 114.4 (-0.9) | 96.53 / 96.90 (+0.37) | 18.7 / 16.4 (-2.3) | 29.13 / 29.55 (+0.42) |
| 524288 | 121.0 / 121.3 (+0.3) | 67702 / 67535 (-168) | 120.3 / 121.7 (+1.4) | 105.27 / 105.52 (+0.25) | 15.7 / 15.8 (+0.1) | 30.91 / 30.96 (+0.05) |
| 786432 | 127.7 / 128.3 (+0.6) | 64150 / 63850 (-300) | 127.4 / 130.7 (+3.3) | 113.87 / 113.93 (+0.06) | 13.8 / 14.4 (+0.6) | 33.83 / 34.31 (+0.48) |
| 1024000 | 132.9 / 133.6 (+0.7) | 61640 / 61317 (-323) | 132.3 / 132.9 (+0.6) | 119.99 / 121.00 (+1.01) | 12.9 / 12.6 (-0.3) | 35.47 / 37.12 (+1.65) |

## 4 x 1792 + 1024 split (lanes4x1792s, test_packed_walls)

| history | steady wall ms off / on (on - off) | steady tok/s off / on (on - off) | resumed wall ms off / on (on - off) | steady device ms off / on (on - off) | steady exposed host ms off / on (on - off) | attention device ms off / on (on - off) |
|---|---|---|---|---|---|---|
| 51200 | 113.1 / 111.0 (-2.1) | 72432 / 73802 (+1370) | 113.2 / 111.3 (-1.9) | 90.26 / 90.78 (+0.52) | 22.8 / 20.2 (-2.6) | 26.61 / 26.92 (+0.31) |
| 131072 | 113.1 / 111.7 (-1.4) | 72432 / 73339 (+908) | 111.1 / 112.8 (+1.7) | 91.62 / 91.69 (+0.07) | 21.5 / 20.0 (-1.5) | 27.91 / 27.85 (-0.06) |
| 139264 | 113.2 / 113.3 (+0.1) | 72368 / 72304 (-64) | 113.9 / 112.8 (-1.1) | 92.96 / 93.00 (+0.04) | 20.2 / 20.3 (+0.1) | 27.75 / 27.79 (+0.04) |
| 262144 | 119.3 / 118.3 (-1.0) | 68667 / 69248 (+580) | 117.9 / 117.6 (-0.3) | 98.07 / 97.88 (-0.19) | 21.2 / 20.4 (-0.8) | 29.70 / 29.98 (+0.28) |
| 524288 | 122.8 / 123.5 (+0.7) | 66710 / 66332 (-378) | 124.6 / 122.6 (-2.0) | 105.45 / 106.68 (+1.23) | 17.4 / 16.8 (-0.6) | 32.50 / 32.34 (-0.16) |
| 786432 | 131.9 / 131.4 (-0.5) | 62108 / 62344 (+236) | 130.5 / 131.4 (+0.9) | 116.84 / 115.90 (-0.94) | 15.1 / 15.5 (+0.4) | 33.99 / 34.75 (+0.76) |
| 1024000 | 136.0 / 135.4 (-0.6) | 60235 / 60502 (+267) | 135.8 / 136.3 (+0.5) | 122.74 / 122.08 (-0.66) | 13.3 / 13.3 (+0.0) | 36.22 / 36.35 (+0.13) |

## Single request, 5120 tokens (test_skip_positions, BRINGUP_CHUNK=5120: 160 rows a chip)

| history | steady wall ms off / on (on - off) | steady tok/s off / on (on - off) | resumed wall ms off / on (on - off) | steady device ms off / on (on - off) | steady exposed host ms off / on (on - off) | attention device ms off / on (on - off) |
|---|---|---|---|---|---|---|
| 51200 | 72.6 / 73.9 (+1.3) | 70523 / 69283 (-1241) | 75.0 / 76.5 (+1.5) | 60.61 / 60.35 (-0.26) | 12.0 / 13.5 (+1.5) | 19.75 / 19.77 (+0.02) |
| 131072 | 74.3 / 76.8 (+2.5) | 68910 / 66667 (-2243) | 76.2 / 77.5 (+1.3) | 62.46 / 62.63 (+0.17) | 11.8 / 14.2 (+2.4) | 19.99 / 20.03 (+0.04) |
| 139264 | 76.0 / 76.2 (+0.2) | 67368 / 67192 (-177) | 77.6 / 78.3 (+0.7) | 62.91 / 62.81 (-0.10) | 13.1 / 13.4 (+0.3) | 19.95 / 20.04 (+0.09) |
| 262144 | 75.7 / 77.0 (+1.3) | 67635 / 66494 (-1142) | 78.3 / 79.3 (+1.0) | 64.68 / 64.54 (-0.14) | 11.0 / 12.5 (+1.5) | 20.37 / 20.10 (-0.27) |
| 524288 | 76.7 / 76.9 (+0.2) | 66754 / 66580 (-174) | 82.0 / 82.0 (+0.0) | 67.96 / 67.72 (-0.24) | 8.7 / 9.2 (+0.5) | 19.99 / 20.04 (+0.05) |
| 786432 | 82.1 / 82.2 (+0.1) | 62363 / 62287 (-76) | 84.2 / 86.4 (+2.2) | 74.33 / 74.17 (-0.16) | 7.8 / 8.0 (+0.2) | 19.93 / 19.79 (-0.14) |
| 1024000 | 83.2 / 83.0 (-0.2) | 61538 / 61687 (+148) | 84.2 / 83.7 (-0.5) | 76.05 / 76.44 (+0.39) | 7.1 / 6.6 (-0.5) | 19.69 / 19.74 (+0.05) |

## test_woa_offon (one load, all 40 layers, DECODER_SKIP=off; H = 139264)

| case | off == on | on repeat | device ms off / on | wo_a ms off / on | host us a wo_a call off / on |
|---|---|---|---|---|---|
| single8192 | 0.0 | 0.0 | 85.414 / 84.815 | 3.058 / 2.656 | 97.4 / 97.8 |
| 4x2k | 0.0 | 0.0 | 94.718 / 94.200 | 3.067 / 2.662 | 109.0 / 116.2 |
| single5120 | 0.0 | 0.0 | 98.413 / 97.734 | 3.062 / 2.651 | 98.8 / 106.7 |

real160 (op level, real inputs, 160-row knobs): off vs on 0.0.
Load: bank copy - interleaved copy 1.97 s for the layers above (interleaved 2.40 s, bank 4.37 s, host prep common 3.34 s).

## Cold no-rebuild (test_no_rebuild, strict)

- off: no_rebuild_pass [1] over 14 paths, program builds after warm-up [0].
- on: no_rebuild_pass [1] over 14 paths, program builds after warm-up [0].
