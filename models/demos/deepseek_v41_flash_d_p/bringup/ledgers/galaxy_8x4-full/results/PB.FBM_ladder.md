# PB.FBM: indexer block maxima on the score's COMPUTE kernel, ladder before / after (galaxy 8x4, 2026-11-22)

Switch: `INDEXER_BMAX_COMPUTE` (call-time; on = the change, off = PB.BM's writer-core 8-key fp32 maxes + the fold).
Thresholds: `INDEXER_BLOCK_SELECT_MIN` / `_MIN_PACKED` = `auto` = each path's measured crossover in keys
(tt/indexer.py `BSEL_MIN_AUTO`): on 163840 (single) / 196608 (packed); off 131072 / never (PB.BM's).

Method (PB.9C's, results/packed_ladder_20261121.md): one executed-layers load (layers 0-20, DECODER_SKIP=decode,
BRINGUP_CHUNK=8192, 1,048,576-token serving slots, agentic_cc text, the resident engram table, every PB switch of the
gate), steady chunk (every segment at history h, mid-request), median of 10 timed walls after warm-up, device ms = one
real-time profiled run (slowest chip), sections = the slowest chip per section. Single 8192 = tests/test_skip_positions.py
(SKIP_POS), chip-aligned 4 x 2k = tests/packed/test_packed_walls.py (LADDER_CASES=4x2k), resident runner, walls first.
Indexer split (3 plain indexers a chunk: layers 2 / 8 / 14 under decode): `indexer` = the score op (+ its block-max
epilogue), `indexer_bmax` = the fold (off) / transpose (on), `indexer_topk` = the top-k(s), `indexer_gather` = the
block gather; 4 x 2k also `indexer_keys` (PB.KR's owned key sends).

## 1. Before / after at the ladder rungs (steady)

before = `INDEXER_BMAX_COMPUTE=off` (PB.BM / PB.GAP defaults: single block path from 131072 keys, packed never);
after = `INDEXER_BMAX_COMPUTE=on` with the auto thresholds (at 139264 both cases stay on the plain top-k: the after
column is the plain-path run there; at 524288 / 1024000 the block path with compute maxes). Section ms in the order
score / block max / top-k / gather.

| case | history (keys) | before wall / device | after wall / device | d wall / d device | before split | after split |
|---|---|---|---|---|---|---|
| single 8192 | 139264 (73728) | 100.3 / 85.53 | 100.5 / 85.80 | +0.2 / +0.3 (same path, noise) | 3.78 / 0 / 1.38 / 0 | 3.78 / 0 / 1.38 / 0 |
| single 8192 | 524288 (266240) | 107.0 / 96.19 | 105.9 / 95.29 | -1.1 / -0.9 | 13.13 / 0.60 / 0.60 / 0.68 | 12.53 / 0.22 / 0.60 / 0.69 |
| single 8192 | 1024000 (516096) | 122.1 / 111.07 | 120.0 / 108.81 | -2.1 / -2.3 | 25.15 / 1.18 / 0.74 / 0.69 | 23.90 / 0.36 / 0.74 / 0.68 |
| 4 x 2k | 139264 (73728) | 110.2 / 92.89 | 111.0 / 93.57 | +0.8 / +0.7 (same path, noise) | 4.56 / 0 / 1.43 / 0 | 4.65 / 0 / 1.43 / 0 |
| 4 x 2k | 524288 (266240) | 125.4 / 110.26 | 122.1 / 107.67 | -3.3 / -2.6 | 12.53 / 0 / 4.91 / 0 | 13.11 / 0.22 / 0.61 / 0.85 |
| 4 x 2k | 1024000 (516096) | 139.5 / 126.76 | 132.8 / 121.23 | -6.7 / -5.5 | 22.90 / 0 / 9.49 / 0 | 23.84 / 0.36 / 0.75 / 0.80 |

The score op with compute maxes vs the plain score (same runs, 3 layers): +0.48 ms at 266240 keys, +0.91 at 516096
(single); the writer path's epilogue was +1.08 / +2.16 (before - plain). The fold (fp32 8-key maxes, 64 MB a chip at
1M) -> the transpose (bf16 32-key maxes, 8 MB): 0.60 -> 0.22, 1.18 -> 0.36 ms.

## 2. Crossovers (block path with compute maxes - plain top-k, steady; resumed in brackets)

| keys (history) | single d device | single d wall | 4 x 2k d device | 4 x 2k d wall |
|---|---|---|---|---|
| 73728 (139264) | -0.19 | +1.9 | +0.38 | +1.3 |
| 135168 (262144) | +0.59 (-1.10) | +1.0 (-1.5) | -1.21 (-1.05) | +2.2 (+2.4) |
| 200704 (393216) | -2.42 (-2.41) | -0.8 (-0.1) | -1.45 (-2.29) | -2.1 (0.0) |
| 266240 (524288) | -2.72 (-3.01) | -2.1 (-1.7) | -2.08 (-3.46) | -2.1 (-2.8) |
| 516096 (1024000) | -7.04 (-6.18) | -9.9 (-6.4) | -5.59 (-4.92) | -8.4 (-7.2) |

Below ~150k keys the block path's extra launches (4 ops a layer: transpose, 2nd top-k, gather, keys) cost more wall than
the top-k saves; the packed resumed wall breaks even only at ~200k. DECIDED: single 163840 keys (the steady wall
crossover ~171k, device ~148k), packed 196608 keys. No ladder rung is slower than block-select off (139264 stays plain
on both). Raw runs: resident logs 20261122-050501 (before), -051109 (block, MIN 0), and the plain / middle-point runs
of the same session (BREADCRUMBS PB.FBM).

## 3. Exactness

- Fork check on the 8x4 (ttnn/ttnn/bringup/indexer_score/tests/unit/test_block_max_compute.py, via
  tests/perf/test_fbm_probe.py): compute maxes == the writer path's folded maxes == torch's max of the same call's
  written bf16 scores, byte for byte, 128 chip cases (32 chips x 4 starts, ratio 2), scores unchanged, nothing written
  past kv_len; the strip transpose byte-exact (check_block_select, 5 cases).
- Model shapes (probe, random data, a 1M slot): maxes of the two paths 0 of 18,874,368 (139264) and 0 of 132,120,576
  (1024000) differ, single ring and the packed keys_gathered ring.
- First version (reduce straight from the fp32 accumulator): 1 bf16 ulp low on ~1 % of the blocks (DS41-389); fixed
  by reducing the bf16 copy of the tile.

## 4. Gate (`gate PB.FBM --check`, final code, the shipped defaults: switch on, auto thresholds)

PASS PB.FBM (devjob fbm_gate; logs run1/logs/PB.FBM.check.shared/). Packed oracle alone vs today 0.0, lane
equivalence 0.0, slot digests 0, packed determinism 0.0; determinism 0.0; cold strict no-rebuild PASS with 0 builds
after warm-up on every path (single8192, serving, one4k, one1600, lanes2x3k, lanes4x2k, lanes4x1600s, lanes6mix), ids
uploads a chunk 0, fallback chunks 0; idx_block_select_set_mismatches 0 (single and packed, 139264 and 1024000).

Ladder from the gate's own units (steady wall ms / device ms; split = score / block max / top-k / gather):

| case | 139264 | 524288 | 1024000 |
|---|---|---|---|
| single 8192 | 100.1 / 86.01 (3.78 / 0 / 1.38 / 0) | 105.6 / 95.24 (12.54 / 0.22 / 0.60 / 0.68) | 119.8 / 109.57 (23.90 / 0.36 / 0.74 / 0.68) |
| 4 x 2k | 111.7 / 94.68 (4.44 / 0 / 1.43 / 0) | 123.4 / 106.83 (13.05 / 0.22 / 0.61 / 0.84) | 134.9 / 121.91 (23.98 / 0.36 / 0.96 / 0.80) |

tok/s: single 81838 / 77576 / 68381; 4 x 2k 73339 / 66386 / 60727. Before (section 1): single 100.3 / 107.0 / 122.1,
4 x 2k 110.2 / 125.4 / 139.5 ms.
