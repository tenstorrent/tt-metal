# QB.1: wq_b on the per-bank C++ grouped_matmul (G = 1), DS41_WQB_BANK off vs on

## Verdict data (2026-11-28, attempt 1; switch on in every gate unit)

- **Accuracy:** the gate's 40-layer reference (test_chunk8192_reference, switch on) PASS: err_layer_L39 0.868 / 0.662
  (limits 1.492 / 1.533), err_final_hidden 0.140 / 0.135 (0.259 / 0.242), top5_overlap 1.0 / 1.0. Determinism 0.0.
- **Off vs on, same load:** bit-identical on every one of 20 layer outputs (single 8192 and packed 4 x 2k, H = 139264),
  though the bank call's K blocking differs from the stock minimal_matmul (DS41-390 allowed a difference; none
  showed, DS41-406). On repeats 0.0 (every layer output and served record).
- **Device win, same load (median of 3 profiled runs a side, alternating; two runs: the gate's and the A/B's):**
  wq_b 5.01 -> 4.13 / 4.14 ms a single 8192 chunk (-0.87 ms, 21 calls, -41 us a call), 4.96-5.00 -> 4.38-4.39 ms a
  4 x 2k chunk (-0.57..-0.62); whole chunk single 8192 84.88 -> 84.00 / 85.07 -> 84.10 ms (-0.89 / -0.97), 4 x 2k
  94.00 -> 93.17 / 94.21 -> 93.43 (-0.83 / -0.77); attention section -0.89 / -0.82 (single), -0.56 / -0.69 (4 x 2k).
  FIX.42's op probe predicted -35.9 us a call (-0.75 ms a chunk).
- **Host a wq_b call** (warm, best of 2): 70.5 -> 41.8 us (single), 78.0 -> 46.3 (4 x 2k): -29..-32 us a call, about
  -0.6 ms host a chunk (the C++ grouped_matmul on its program cache vs minimal_matmul's launch).
- **Weights:** shipped load (WQB_BANK_COPIES=selected) holds one wq_b copy (42.5 MiB bfp8 a layer a chip), the same
  bytes either layout; the off / on test adds the other copy for its run and drops it after (21 layers in 4.4 s).
- **No rebuild:** strict cold run, switch on, every packed path 0 program builds after warm-up.
- **Ladders:** the tables below are the switch-on side; the off vs on ladder table is the orchestrator's A/B (not run
  by this step).

## Same load, off vs on (test_wqb_offon, layers 0-20, DECODER_SKIP=decode, H = 139264)

| case | device ms off | on | attention ms off | on | wq_b ms off | on | host us a wq_b call off | on | on repeat | off vs on max rel L2 a layer |
|---|---|---|---|---|---|---|---|---|---|---|
| single8192 | 85.07 | 84.10 | 26.33 | 25.51 | 5.01 | 4.13 | 69.1 | 42.1 | 0.0 | 0.0e+00 (20/20 layers bit-identical) |
| 4x2k | 94.21 | 93.43 | 24.94 | 24.25 | 5.00 | 4.38 | 76.2 | 47.2 | 0.0 | 0.0e+00 (20/20 layers bit-identical) |

Runs a side (device ms): single8192 off [85.135, 84.879, 85.073] on [84.103, 84.095, 84.249]; 4x2k off [93.738, 94.207, 94.479] on [91.274, 93.433, 94.274]

## Ladders, switch on (the gate's units; the off side is the orchestrator's A/B)

Single 8192 (test_skip_positions):

| H | steady wall ms | tok/s | resumed wall ms | device ms | attention ms | exposed host ms |
|---|---|---|---|---|---|---|
| 51200 | 96.30 | 85068 | 98.20 | 81.81 | 26.12 | 14.50 |
| 131072 | 95.10 | 86141 | 99.50 | 83.58 | 26.10 | 11.50 |
| 139264 | 94.90 | 86322 | 98.30 | 83.09 | 26.07 | 11.80 |
| 262144 | 98.80 | 82915 | 102.80 | 88.56 | 26.14 | 10.20 |
| 524288 | 103.30 | 79303 | 106.10 | 92.42 | 26.18 | 10.90 |
| 786432 | 113.30 | 72304 | 116.90 | 102.23 | 26.14 | 11.10 |
| 1024000 | 118.40 | 69189 | 120.90 | 106.77 | 26.50 | 11.60 |

Packed 4x2k (test_packed_walls):

| H | steady wall ms | tok/s | device ms | attention ms | exposed host ms |
|---|---|---|---|---|---|
| 51200 | 104.50 | 78392 | 89.23 | 25.89 | 15.30 |
| 131072 | 104.20 | 78618 | 89.74 | 26.88 | 14.50 |
| 139264 | 105.40 | 77723 | 90.87 | 27.16 | 14.50 |
| 262144 | 107.50 | 76205 | 95.89 | 29.10 | 11.60 |
| 524288 | 115.20 | 71111 | 105.52 | 31.76 | 9.70 |
| 786432 | 124.70 | 65694 | 116.18 | 37.24 | 8.50 |
| 1024000 | 131.20 | 62439 | 120.43 | 35.25 | 10.80 |

Packed 4x1792s (test_packed_walls):

| H | steady wall ms | tok/s | device ms | attention ms | exposed host ms |
|---|---|---|---|---|---|
| 51200 | 106.20 | 77138 | 88.94 | 26.02 | 17.30 |
| 131072 | 106.20 | 77138 | 90.47 | 27.00 | 15.70 |
| 139264 | 108.70 | 75363 | 92.11 | 27.18 | 16.60 |
| 262144 | 110.50 | 74136 | 97.07 | 28.99 | 13.40 |
| 524288 | 117.00 | 70017 | 104.76 | 31.58 | 12.20 |
| 786432 | 127.30 | 64352 | 115.80 | 34.46 | 11.50 |
| 1024000 | 135.00 | 60682 | 123.67 | 36.12 | 11.30 |

No-rebuild (strict, cold, switch on): 30 metrics, pass True; determinism_max_abs_diff 0.0.
