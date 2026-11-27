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
- **Ladders:** the tables below, off = WA.4R / WA.4R5 (switch off, own loads), on = the A/B's new side. Single 8192
  attention -0.47 ms (median over rungs), 5120 -0.34; packed device deltas sit within the loads' drift.
- **Load time:** the bank copy costs +1.97 s over the interleaved copy for all 40 layers (bank 4.37 s, interleaved
  2.40 s, host prep 3.34 s common to both); WA.3's 21-layer number was +1.39 s. Not cached.
- **Weight DRAM a chip, wo_a:** 35.65 MB a layer. Shipped load (DS41_WOA_BANK_COPIES=selected, OV-77): one copy,
  1.33 GiB for 40 layers (0.70 GiB for the 21-layer perf load): the same as today, whichever layout is selected. A/B /
  test load (both): 2.66 GiB for 40 (+1.33 GiB), only while an off / on test holds both (test_woa_offon adds the copy and
  drops it after). The ladders' own `dram_load_gib_max` (perf load, held by the resident runner): 10.75 GiB (8192),
  9.16 GiB (5120).
- **Cold no-rebuild (strict, fresh process, switch on):** PASS on every path, 0 program builds after warm-up.

**Off side of the ladders (re-rendered at FIX.40, F308):** the WA.4 A/B's off side ran every ladder with
DS41_WOA_BANK=on (each gate command's own prefix won over the side's env; DS41-397), so the off columns below are now
the gate results of tasks WA.4R (single 8192, 4 x 2k, 4 x 1792s; one fresh load, DS41_WOA_BANK=off) and WA.4R5
(single 5120, its own load, off); the on columns are the A/B's new side. Off and on are different loads (walls and
exposed host drift 1-3 ms between loads: judge device ms). on - off: single 8192 attention -0.88..-0.23 ms (median
-0.47) at every rung, device -0.73..+0.08 (median -0.39); single 5120 attention -0.45..-0.26 (median -0.34), device
-0.66..-0.02 (median -0.31); packed 4 x 2k device -1.25..+1.05 (median +0.42), attention -0.46..+1.21 (median -0.02);
4 x 1792s device -0.44..+0.36 (median +0.03), attention -0.37..+0.61 (median -0.16). The single-request ladders show
the same-load win (test_woa_offon -0.40 ms wo_a a chunk; the op probe's -0.35 ms at 5120); the packed ladders' device
drift between loads (±1.2 ms) is larger than the win, so they neither confirm nor refute it. The same-load verdict
stays test_woa_offon (wo_a 3.058 -> 2.656 ms, chunk 85.41 -> 84.82 ms single 8192; 94.72 -> 94.20 ms 4 x 2k).

Rendered by tests/woa/woa_ladder_report.py from the A/B runs of WA.4 (/scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/runs/galaxy_8x4-full/run1/ab/WA.4; new = on, off = tasks WA.4R / WA.4R5,
the gate's commands, one resident load a side's chunk size). Ladder method as PB.LAD
(tests/packed/ladder_common.py): steady = mid-request chunk, resumed = first chunk after a prefix hit; 10 timed
runs, median, program-cache misses refused; device ms = one profiled run (slowest chip); exposed host = wall -
device. Layers 0-20 (DECODER_SKIP=decode). `-` = that side has not run (yet).

**Off side = tasks WA.4R / WA.4R5 (FIX.40, F308):** the off columns of the ladder tables come from the gate results results/WA.4R.json (single 8192, 4 x 2k, 4 x 1792s: one fresh load, DS41_WOA_BANK=off) and results/WA.4R5.json (single 5120, its own load, DS41_WOA_BANK=off); the on columns are the A/B's new side. **Off and on were measured in different loads: walls and exposed host drift between loads by 1-3 ms; judge device ms (and the attention section, where wo_a sits).** The A/B's own old side (switch on, F308) is not shown; test_no_rebuild was not re-run with the switch off (its off line below ran with it on).

## Single request, 8192 tokens (test_skip_positions, BRINGUP_CHUNK=8192)

| history | steady wall ms off / on (on - off) | steady tok/s off / on (on - off) | resumed wall ms off / on (on - off) | steady device ms off / on (on - off) | steady exposed host ms off / on (on - off) | attention device ms off / on (on - off) |
|---|---|---|---|---|---|---|
| 51200 | 99.0 / 99.9 (+0.9) | 82748 / 82002 (-746) | 101.6 / 101.8 (+0.2) | 82.78 / 82.39 (-0.39) | 16.2 / 17.5 (+1.3) | 27.48 / 27.00 (-0.48) |
| 131072 | 98.9 / 98.0 (-0.9) | 82831 / 83592 (+761) | 103.2 / 102.7 (-0.5) | 84.72 / 84.20 (-0.52) | 14.2 / 13.8 (-0.4) | 27.34 / 26.81 (-0.53) |
| 139264 | 96.8 / 98.1 (+1.3) | 84628 / 83507 (-1122) | 102.5 / 102.6 (+0.1) | 84.52 / 83.94 (-0.58) | 12.3 / 14.2 (+1.9) | 27.32 / 26.86 (-0.46) |
| 262144 | 100.9 / 100.3 (-0.6) | 81189 / 81675 (+486) | 105.1 / 103.7 (-1.4) | 89.30 / 89.38 (+0.08) | 11.6 / 10.9 (-0.7) | 27.23 / 27.00 (-0.23) |
| 524288 | 104.7 / 103.7 (-1.0) | 78243 / 78997 (+754) | 108.9 / 108.8 (-0.1) | 93.62 / 93.51 (-0.11) | 11.1 / 10.2 (-0.9) | 27.40 / 27.02 (-0.38) |
| 786432 | 116.0 / 114.4 (-1.6) | 70621 / 71608 (+988) | 119.4 / 118.8 (-0.6) | 104.62 / 103.89 (-0.73) | 11.4 / 10.5 (-0.9) | 27.94 / 27.06 (-0.88) |
| 1024000 | 120.2 / 120.3 (+0.1) | 68153 / 68096 (-57) | 123.8 / 124.0 (+0.2) | 108.25 / 107.95 (-0.30) | 12.0 / 12.3 (+0.3) | 27.89 / 27.42 (-0.47) |

on - off over the rungs: device -0.73..+0.08 ms (median -0.39); attention -0.88..-0.23 ms (median -0.47).

## Chip-aligned 4 x 2048 (lanes4x2k, test_packed_walls)

| history | steady wall ms off / on (on - off) | steady tok/s off / on (on - off) | resumed wall ms off / on (on - off) | steady device ms off / on (on - off) | steady exposed host ms off / on (on - off) | attention device ms off / on (on - off) |
|---|---|---|---|---|---|---|
| 51200 | 110.0 / 108.4 (-1.6) | 74473 / 75572 (+1099) | 108.1 / 107.8 (-0.3) | 91.29 / 90.08 (-1.21) | 18.7 / 18.3 (-0.4) | 27.04 / 26.88 (-0.16) |
| 131072 | 109.7 / 110.2 (+0.5) | 74676 / 74338 (-339) | 108.4 / 108.3 (-0.1) | 91.07 / 91.53 (+0.46) | 18.6 / 18.7 (+0.1) | 27.99 / 28.10 (+0.11) |
| 139264 | 113.1 / 109.5 (-3.6) | 72432 / 74813 (+2381) | 110.2 / 110.8 (+0.6) | 92.18 / 92.07 (-0.11) | 20.9 / 17.4 (-3.5) | 28.05 / 27.59 (-0.46) |
| 262144 | 114.8 / 113.3 (-1.5) | 71359 / 72304 (+945) | 114.8 / 114.4 (-0.4) | 96.48 / 96.90 (+0.42) | 18.3 / 16.4 (-1.9) | 29.37 / 29.55 (+0.18) |
| 524288 | 122.1 / 121.3 (-0.8) | 67092 / 67535 (+442) | 122.1 / 121.7 (-0.4) | 104.56 / 105.52 (+0.96) | 17.5 / 15.8 (-1.7) | 30.98 / 30.96 (-0.02) |
| 786432 | 129.3 / 128.3 (-1.0) | 63356 / 63850 (+494) | 129.0 / 130.7 (+1.7) | 115.18 / 113.93 (-1.25) | 14.1 / 14.4 (+0.3) | 34.63 / 34.31 (-0.32) |
| 1024000 | 133.0 / 133.6 (+0.6) | 61594 / 61317 (-277) | 134.2 / 132.9 (-1.3) | 119.95 / 121.00 (+1.05) | 13.1 / 12.6 (-0.5) | 35.91 / 37.12 (+1.21) |

on - off over the rungs: device -1.25..+1.05 ms (median +0.42); attention -0.46..+1.21 ms (median -0.02).

## 4 x 1792 + 1024 split (lanes4x1792s, test_packed_walls)

| history | steady wall ms off / on (on - off) | steady tok/s off / on (on - off) | resumed wall ms off / on (on - off) | steady device ms off / on (on - off) | steady exposed host ms off / on (on - off) | attention device ms off / on (on - off) |
|---|---|---|---|---|---|---|
| 51200 | 111.8 / 111.0 (-0.8) | 73274 / 73802 (+528) | 112.3 / 111.3 (-1.0) | 90.53 / 90.78 (+0.25) | 21.3 / 20.2 (-1.1) | 27.12 / 26.92 (-0.20) |
| 131072 | 115.1 / 111.7 (-3.4) | 71173 / 73339 (+2166) | 112.0 / 112.8 (+0.8) | 92.13 / 91.69 (-0.44) | 23.0 / 20.0 (-3.0) | 28.09 / 27.85 (-0.24) |
| 139264 | 112.9 / 113.3 (+0.4) | 72560 / 72304 (-256) | 113.8 / 112.8 (-1.0) | 93.37 / 93.00 (-0.37) | 19.5 / 20.3 (+0.8) | 27.95 / 27.79 (-0.16) |
| 262144 | 119.6 / 118.3 (-1.3) | 68495 / 69248 (+753) | 118.4 / 117.6 (-0.8) | 97.85 / 97.88 (+0.03) | 21.7 / 20.4 (-1.3) | 29.98 / 29.98 (+0.00) |
| 524288 | 124.8 / 123.5 (-1.3) | 65641 / 66332 (+691) | 123.0 / 122.6 (-0.4) | 106.89 / 106.68 (-0.21) | 17.9 / 16.8 (-1.1) | 32.71 / 32.34 (-0.37) |
| 786432 | 130.3 / 131.4 (+1.1) | 62870 / 62344 (-526) | 129.7 / 131.4 (+1.7) | 115.64 / 115.90 (+0.26) | 14.7 / 15.5 (+0.8) | 34.14 / 34.75 (+0.61) |
| 1024000 | 137.1 / 135.4 (-1.7) | 59752 / 60502 (+750) | 136.6 / 136.3 (-0.3) | 121.72 / 122.08 (+0.36) | 15.4 / 13.3 (-2.1) | 35.85 / 36.35 (+0.50) |

on - off over the rungs: device -0.44..+0.36 ms (median +0.03); attention -0.37..+0.61 ms (median -0.16).

## Single request, 5120 tokens (test_skip_positions, BRINGUP_CHUNK=5120: 160 rows a chip)

| history | steady wall ms off / on (on - off) | steady tok/s off / on (on - off) | resumed wall ms off / on (on - off) | steady device ms off / on (on - off) | steady exposed host ms off / on (on - off) | attention device ms off / on (on - off) |
|---|---|---|---|---|---|---|
| 51200 | 73.1 / 73.9 (+0.8) | 70041 / 69283 (-758) | 76.6 / 76.5 (-0.1) | 60.66 / 60.35 (-0.31) | 12.4 / 13.5 (+1.1) | 20.11 / 19.77 (-0.34) |
| 131072 | 75.5 / 76.8 (+1.3) | 67815 / 66667 (-1148) | 77.4 / 77.5 (+0.1) | 62.96 / 62.63 (-0.33) | 12.5 / 14.2 (+1.7) | 20.29 / 20.03 (-0.26) |
| 139264 | 76.2 / 76.2 (+0.0) | 67192 / 67192 (+0) | 79.8 / 78.3 (-1.5) | 63.08 / 62.81 (-0.27) | 13.1 / 13.4 (+0.3) | 20.32 / 20.04 (-0.28) |
| 262144 | 75.9 / 77.0 (+1.1) | 67457 / 66494 (-964) | 78.2 / 79.3 (+1.1) | 64.99 / 64.54 (-0.45) | 10.9 / 12.5 (+1.6) | 20.52 / 20.10 (-0.42) |
| 524288 | 77.7 / 76.9 (-0.8) | 65894 / 66580 (+686) | 81.9 / 82.0 (+0.1) | 68.38 / 67.72 (-0.66) | 9.3 / 9.2 (-0.1) | 20.35 / 20.04 (-0.31) |
| 786432 | 81.9 / 82.2 (+0.3) | 62515 / 62287 (-228) | 84.8 / 86.4 (+1.6) | 74.34 / 74.17 (-0.17) | 7.6 / 8.0 (+0.4) | 20.15 / 19.79 (-0.36) |
| 1024000 | 85.6 / 83.0 (-2.6) | 59813 / 61687 (+1874) | 85.2 / 83.7 (-1.5) | 76.46 / 76.44 (-0.02) | 9.1 / 6.6 (-2.5) | 20.19 / 19.74 (-0.45) |

on - off over the rungs: device -0.66..-0.02 ms (median -0.31); attention -0.45..-0.26 ms (median -0.34).

## test_woa_offon (one load, all 40 layers, DECODER_SKIP=off; H = 139264)

| case | off == on | on repeat | device ms off / on | wo_a ms off / on | host us a wo_a call off / on |
|---|---|---|---|---|---|
| single8192 | 0.0 | 0.0 | 85.414 / 84.815 | 3.058 / 2.656 | 97.4 / 97.8 |
| 4x2k | 0.0 | 0.0 | 94.718 / 94.200 | 3.067 / 2.662 | 109.0 / 116.2 |
| single5120 | 0.0 | 0.0 | 98.413 / 97.734 | 3.062 / 2.651 | 98.8 / 106.7 |

real160 (op level, real inputs, 160-row knobs): off vs on 0.0.
Load: bank copy - interleaved copy 1.97 s for the layers above (interleaved 2.40 s, bank 4.37 s, host prep common 3.34 s).

## Cold no-rebuild (test_no_rebuild, strict)

- A/B old side (ran with switch on, F308): no_rebuild_pass [1] over 14 paths, program builds after warm-up [0].
- on: no_rebuild_pass [1] over 14 paths, program builds after warm-up [0].
