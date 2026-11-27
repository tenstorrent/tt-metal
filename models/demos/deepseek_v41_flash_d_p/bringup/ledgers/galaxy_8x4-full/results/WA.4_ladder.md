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

**The off side of the ladders is missing from this A/B:** `orchestrator measure --task WA.4` stops in
orchestrator.ab_batches -> metrics.reset with `OSError: [Errno 36] File name too long`: ab_test_names names the
single-8192 test_skip_positions unit after every variable it does not share with the 5120 unit (BRINGUP_CHUNK and 11
DS41_PACKED* / POS_DYNAMIC switches: 316 characters, a file name allows 255). This is framework code (outside this step's paths), and the
orchestrator's own A/B after this step goes through the same code. The on side was measured per part
(test_woa_offon, test_packed_walls, test_skip_positions__BRINGUP_CHUNK-5120, test_no_rebuild) and the single-8192
ladder through the gate unit WA.4#test_skip_positions. The off columns below are therefore a labelled PROXY (the last
accepted ladder, PB.MM256's on side: the off path's code before WA.3). Once the naming is fixed and the A/B has run
both sides, `python3 -m models.demos.deepseek_v41_flash_d_p.tests.woa.woa_ladder_report` renders the real off vs on
tables (this section: tests/woa/wa4_notes.md).

Rendered by tests/woa/woa_ladder_report.py from the A/B runs of WA.4 (/scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/runs/galaxy_8x4-full/run1/ab/WA.4; old = off, new = on,
the gate's commands, one resident load a side's chunk size, the switch call-time). Ladder method as PB.LAD
(tests/packed/ladder_common.py): steady = mid-request chunk, resumed = first chunk after a prefix hit; 10 timed
runs, median, program-cache misses refused; device ms = one profiled run (slowest chip); exposed host = wall -
device. Layers 0-20 (DECODER_SKIP=decode). `-` = that side has not run (yet).

**Off side of single 8192, 4x2k / 4x1792s: a PROXY** (/scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/runs/galaxy_8x4-full/run1/ab/PB.MM256/new: another session's runs of the off path's code,
not this A/B's; walls drift a few ms between sessions, DS41-385). Read its deltas as indicative only.

## Single request, 8192 tokens (test_skip_positions, BRINGUP_CHUNK=8192)

| history | steady wall ms off / on (on - off) | steady tok/s off / on (on - off) | resumed wall ms off / on (on - off) | steady device ms off / on (on - off) | steady exposed host ms off / on (on - off) | attention device ms off / on (on - off) |
|---|---|---|---|---|---|---|
| 51200 | 98.7 / 97.1 (-1.6) | 82999 / 84367 (+1368) | 101.7 / 102.6 (+0.9) | 82.73 / 82.37 (-0.36) | 16.0 / 14.7 (-1.3) | 27.51 / 27.06 (-0.45) |
| 131072 | 100.0 / 101.1 (+1.1) | 81920 / 81029 (-891) | 103.3 / 102.9 (-0.4) | 84.86 / 84.53 (-0.33) | 15.1 / 16.6 (+1.5) | 27.36 / 26.84 (-0.52) |
| 139264 | 98.8 / 98.6 (-0.2) | 82915 / 83083 (+168) | 101.4 / 102.8 (+1.4) | 84.62 / 83.73 (-0.89) | 14.2 / 14.9 (+0.7) | 27.29 / 26.92 (-0.37) |
| 262144 | 100.5 / 102.4 (+1.9) | 81512 / 80000 (-1512) | 105.9 / 105.4 (-0.5) | 89.44 / 89.25 (-0.19) | 11.1 / 13.1 (+2.0) | 27.27 / 26.93 (-0.34) |
| 524288 | 106.6 / 106.8 (+0.2) | 76848 / 76704 (-144) | 109.6 / 108.4 (-1.2) | 94.29 / 93.93 (-0.36) | 12.3 / 12.9 (+0.6) | 27.41 / 27.16 (-0.25) |
| 786432 | 115.4 / 115.8 (+0.4) | 70988 / 70743 (-245) | 121.8 / 120.4 (-1.4) | 104.15 / 103.84 (-0.31) | 11.3 / 12.0 (+0.7) | 27.47 / 27.06 (-0.41) |
| 1024000 | 120.2 / 120.3 (+0.1) | 68153 / 68096 (-57) | 123.3 / 123.0 (-0.3) | 108.25 / 107.44 (-0.81) | 12.0 / 12.9 (+0.9) | 27.85 / 27.32 (-0.53) |

## Chip-aligned 4 x 2048 (lanes4x2k, test_packed_walls)

| history | steady wall ms off / on (on - off) | steady tok/s off / on (on - off) | resumed wall ms off / on (on - off) | steady device ms off / on (on - off) | steady exposed host ms off / on (on - off) | attention device ms off / on (on - off) |
|---|---|---|---|---|---|---|
| 51200 | 109.2 / 108.4 (-0.8) | 75018 / 75572 (+554) | 108.3 / 107.8 (-0.5) | 89.40 / 90.08 (+0.68) | 19.8 / 18.3 (-1.5) | 27.14 / 26.88 (-0.26) |
| 131072 | 109.7 / 110.2 (+0.5) | 74676 / 74338 (-339) | 110.3 / 108.3 (-2.0) | 91.16 / 91.53 (+0.37) | 18.5 / 18.7 (+0.2) | 27.98 / 28.10 (+0.12) |
| 139264 | 111.9 / 109.5 (-2.4) | 73208 / 74813 (+1605) | 110.9 / 110.8 (-0.1) | 92.17 / 92.07 (-0.10) | 19.7 / 17.4 (-2.3) | 28.28 / 27.59 (-0.69) |
| 262144 | 115.3 / 113.3 (-2.0) | 71049 / 72304 (+1254) | 114.3 / 114.4 (+0.1) | 96.35 / 96.90 (+0.55) | 19.0 / 16.4 (-2.6) | 29.20 / 29.55 (+0.35) |
| 524288 | 122.0 / 121.3 (-0.7) | 67148 / 67535 (+388) | 121.2 / 121.7 (+0.5) | 106.29 / 105.52 (-0.77) | 15.7 / 15.8 (+0.1) | 31.39 / 30.96 (-0.43) |
| 786432 | 127.4 / 128.3 (+0.9) | 64301 / 63850 (-451) | 126.7 / 130.7 (+4.0) | 113.90 / 113.93 (+0.03) | 13.5 / 14.4 (+0.9) | 33.65 / 34.31 (+0.66) |
| 1024000 | 133.7 / 133.6 (-0.1) | 61272 / 61317 (+46) | 132.3 / 132.9 (+0.6) | 120.42 / 121.00 (+0.58) | 13.3 / 12.6 (-0.7) | 36.35 / 37.12 (+0.77) |

## 4 x 1792 + 1024 split (lanes4x1792s, test_packed_walls)

| history | steady wall ms off / on (on - off) | steady tok/s off / on (on - off) | resumed wall ms off / on (on - off) | steady device ms off / on (on - off) | steady exposed host ms off / on (on - off) | attention device ms off / on (on - off) |
|---|---|---|---|---|---|---|
| 51200 | 112.0 / 111.0 (-1.0) | 73143 / 73802 (+659) | 112.9 / 111.3 (-1.6) | 90.55 / 90.78 (+0.23) | 21.5 / 20.2 (-1.3) | 27.03 / 26.92 (-0.11) |
| 131072 | 113.9 / 111.7 (-2.2) | 71923 / 73339 (+1417) | 113.1 / 112.8 (-0.3) | 92.05 / 91.69 (-0.36) | 21.8 / 20.0 (-1.8) | 27.98 / 27.85 (-0.13) |
| 139264 | 113.9 / 113.3 (-0.6) | 71923 / 72304 (+381) | 114.4 / 112.8 (-1.6) | 94.70 / 93.00 (-1.70) | 19.2 / 20.3 (+1.1) | 28.53 / 27.79 (-0.74) |
| 262144 | 120.1 / 118.3 (-1.8) | 68210 / 69248 (+1038) | 118.4 / 117.6 (-0.8) | 98.04 / 97.88 (-0.16) | 22.1 / 20.4 (-1.7) | 29.96 / 29.98 (+0.02) |
| 524288 | 126.2 / 123.5 (-2.7) | 64913 / 66332 (+1419) | 123.6 / 122.6 (-1.0) | 106.55 / 106.68 (+0.13) | 19.6 / 16.8 (-2.8) | 32.87 / 32.34 (-0.53) |
| 786432 | 132.1 / 131.4 (-0.7) | 62014 / 62344 (+330) | 131.4 / 131.4 (+0.0) | 115.89 / 115.90 (+0.01) | 16.2 / 15.5 (-0.7) | 34.92 / 34.75 (-0.17) |
| 1024000 | 135.9 / 135.4 (-0.5) | 60280 / 60502 (+223) | 136.8 / 136.3 (-0.5) | 123.13 / 122.08 (-1.05) | 12.8 / 13.3 (+0.5) | 35.71 / 36.35 (+0.64) |

## Single request, 5120 tokens (test_skip_positions, BRINGUP_CHUNK=5120: 160 rows a chip)

| history | steady wall ms off / on (on - off) | steady tok/s off / on (on - off) | resumed wall ms off / on (on - off) | steady device ms off / on (on - off) | steady exposed host ms off / on (on - off) | attention device ms off / on (on - off) |
|---|---|---|---|---|---|---|
| 51200 | - / 73.9 (-) | - / 69283 (-) | - / 76.5 (-) | - / 60.35 (-) | - / 13.5 (-) | - / 19.77 (-) |
| 131072 | - / 76.8 (-) | - / 66667 (-) | - / 77.5 (-) | - / 62.63 (-) | - / 14.2 (-) | - / 20.03 (-) |
| 139264 | - / 76.2 (-) | - / 67192 (-) | - / 78.3 (-) | - / 62.81 (-) | - / 13.4 (-) | - / 20.04 (-) |
| 262144 | - / 77.0 (-) | - / 66494 (-) | - / 79.3 (-) | - / 64.54 (-) | - / 12.5 (-) | - / 20.10 (-) |
| 524288 | - / 76.9 (-) | - / 66580 (-) | - / 82.0 (-) | - / 67.72 (-) | - / 9.2 (-) | - / 20.04 (-) |
| 786432 | - / 82.2 (-) | - / 62287 (-) | - / 86.4 (-) | - / 74.17 (-) | - / 8.0 (-) | - / 19.79 (-) |
| 1024000 | - / 83.0 (-) | - / 61687 (-) | - / 83.7 (-) | - / 76.44 (-) | - / 6.6 (-) | - / 19.74 (-) |

## test_woa_offon (one load, all 40 layers, DECODER_SKIP=off; H = 139264)

| case | off == on | on repeat | device ms off / on | wo_a ms off / on | host us a wo_a call off / on |
|---|---|---|---|---|---|
| single8192 | 0.0 | 0.0 | 85.414 / 84.815 | 3.058 / 2.656 | 97.4 / 97.8 |
| 4x2k | 0.0 | 0.0 | 94.718 / 94.200 | 3.067 / 2.662 | 109.0 / 116.2 |
| single5120 | 0.0 | 0.0 | 98.413 / 97.734 | 3.062 / 2.651 | 98.8 / 106.7 |

real160 (op level, real inputs, 160-row knobs): off vs on 0.0.
Load: bank copy - interleaved copy 1.97 s for the layers above (interleaved 2.40 s, bank 4.37 s, host prep common 3.34 s).

## Cold no-rebuild (test_no_rebuild, strict)

- off: no_rebuild_pass - over 0 paths, program builds after warm-up -.
- on: no_rebuild_pass [1] over 14 paths, program builds after warm-up [0].
