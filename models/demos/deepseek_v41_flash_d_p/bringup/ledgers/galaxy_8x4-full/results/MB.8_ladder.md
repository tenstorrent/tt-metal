# MB.8: compute-kernel indexer block maxes (INDEXER_BMAX_COMPUTE on | off) on the single 5120 path

8x4 Galaxy, chunk 5120, agentic_cc text, DECODER_SKIP=decode, resident engram table. on = PB.FBM's block maxes on the
score op's compute kernel + the strip transpose; off = PB.BM's writer-core 8-key maxes + the fold (MB.7's code).
INDEXER_BLOCK_SELECT_MIN = auto = 262144 keys on both sides, so 51200 .. 262144 run the plain top-k on both sides
(the same code) and 524288+ take the block path.

## 1. Crossover with the compute maxes (block vs plain forced, one load, in-process)

tests/perf/test_mb6_bsel_crossover.py, DS41_INDEXER_BMAX_COMPUTE=on. Runs 1-2: 10 timed runs a side; runs 3-4: 20.
block - plain, ms. Wall: each run's median, and the median of all timed runs pooled; device: mean over runs.

| positions | keys | wall per run | wall pooled | device | MB.7 (off) wall / device, two runs |
|---|---|---|---|---|---|
| 131072 | 68096 | -1.53, +4.28 | +3.06 | -0.08 | - |
| 196608 | 100864 | +0.27, +0.10 | +0.46 | -1.05 | - |
| 262144 | 133632 | +1.42, +1.34 | +1.31 | -0.71 | +1.81, +1.99 / -0.72, -0.34 |
| 393216 | 199168 | -0.64, +1.22, +0.06, +0.86 | +0.35 | -1.25 | +2.65, +2.03 / -1.01, -0.93 |
| 524288 | 264704 | +1.59, -0.69, -1.06, +0.31 | -0.16 | -2.28 | -0.53, -0.43 / -1.48, -1.54 |
| 655360 | 330240 | +0.04, -1.12 | -0.50 | -2.29 | - |
| 786432 | 395776 | -1.69, -2.56 | -2.09 | -3.16 | - |

The block path is cheaper on the device at every point (and cheaper than with the writer maxes), but the chunk is
host-bound there (wall - device ~11-14 ms) and the block path's 12 extra launches are unchanged, so the wall crossover
stays between 199168 and 264704 keys. Decided: on = 262144 keys (the smallest count above which no point is slower in
pooled wall). Per run, 264704 keys was slower in 2 of 4 runs; only 395776 keys was faster in every run (owner's call:
DS41_INDEXER_BLOCK_SELECT_MIN=393216 would put the 524288 rung on plain).

## 2. Ladder, on (gate MB.8, 2026-11-22 06:23) vs MB.7's gate (the off code, 2026-11-22 02:59): separate processes

Steady chunk, ms (wall: median of 10 warm runs). Cross-process: wall noise +-1-2 ms (spreads up to 20 ms at
131072 / 139264); the orchestrator's same-run on / off A/B is the deciding table.

| position | wall off | wall on | device off | device on | host off | host on | idx bmax off | idx bmax on | idx score off | idx score on | tok/s off | tok/s on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 51200 | 73.7 | 74.1 | 60.62 | 60.28 | 13.1 | 13.8 | - | - | 1.09 | 1.08 | 69471 | 69096 |
| 131072 | 75.2 | 75.4 | 62.53 | 62.12 | 12.7 | 13.3 | - | - | 1.99 | 1.98 | 68085 | 67905 |
| 139264 | 75.6 | 77.4 | 62.75 | 62.80 | 12.9 | 14.6 | - | - | 2.15 | 2.08 | 67725 | 66150 |
| 262144 | 76.8 | 76.8 | 64.43 | 64.79 | 12.4 | 12.0 | - | - | 3.46 | 3.48 | 66667 | 66667 |
| 524288 | 79.4 | 78.0 | 67.58 | 67.54 | 11.8 | 10.5 | 0.41 | 0.15 | 6.68 | 6.79 | 64484 | 65641 |
| 786432 | 82.2 | 81.3 | 74.76 | 74.22 | 7.4 | 7.1 | 0.60 | 0.22 | 9.69 | 9.79 | 62287 | 62977 |
| 1024000 | 82.9 | 83.1 | 76.09 | 75.61 | 6.8 | 7.5 | 0.75 | 0.22 | 12.45 | 12.63 | 61761 | 61613 |

Rungs <= 262144 run the same code on both sides (differences = noise). On the block rungs the fold -> transpose
saves 0.26-0.53 ms a chunk (3 layers) and the score section grows 0.10-0.18 ms (the compute reduce + the 64-byte
writes), net device -0.04 / -0.54 / -0.48 ms at 524288 / 786432 / 1024000; programs a chunk unchanged (24800 / 23520 /
22240). Below the estimate (-1.3..-1.5 ms at 1M): at 160 query rows a chip the writer's 8-key max cost the score
little, so moving it to the FPU saved less than PB.FBM's 256-row chunk.

## 3. Accuracy and serving (gate MB.8 --check, change on)

PASS: idx_block_select_set_mismatches 0 (single 138240 / 1018880), determinism 0.0, contract 0 / decoder-skip 0
failed, outside history 0.0, program builds after warm-up 0 (cold), ids uploads 0, err_final_hidden 0.136 <= 0.242,
top5 1.0. Fork byte checks on the 8x4 (resident mesh): check_block_max_compute 128 chip cases exact,
check_block_select 15 cases exact.
