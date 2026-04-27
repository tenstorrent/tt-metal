# Structured Output Bitmask Overhead Ablation Report

## Test Configuration
- Model: Llama-3.3-70B-Instruct on TG (Galaxy, 8x4 mesh)
- Data parallel: 4, max_num_seqs: 8 per DP rank
- Bitmask transfer: synchronous (copy + synchronize_device)
- 5 runs per configuration, fresh server restart and device reset per ablation level
- All levels use `benchmark_serving_structured_output.py` with
  `--dataset json-unique --output-len 1000` (same prompts/output distribution)
- L0 adds `--no-structured-output` to disable grammar constraints
  while keeping the same workload

## Ablation Levels

| Level | What runs | What's skipped |
|-------|-----------|----------------|
| L0 | Same prompts, no grammar constraints | Everything |
| L1 | Grammar FSM update (scheduler) | Reorder, gather, transfer, decompress, apply |
| L2 | L1 + reorder to batch order + DP gather | Transfer, decompress, apply |
| L3 | L2 + host-to-device transfer (copy + sync) | Decompress, apply |
| L4 | Full pipeline | Nothing |

Gate controlled by `TT_BITMASK_ABLATION` env var (1-4). Each level
includes all work from previous levels and adds one more step.

## ITL (Inter-Token Latency)

### Mean ITL (ms)

| Level | np=32 | np=64 | np=128 |
|-------|-------|-------|--------|
| L0: No structured output | 20.45 ± 1.73 | 26.77 ± 1.37 | 32.47 ± 1.65 |
| L1: Grammar FSM only | 22.50 ± 1.66 | 28.65 ± 1.25 | 31.82 ± 1.00 |
| L2: + Reorder + DP Gather | 23.15 ± 1.64 | 27.41 ± 0.61 | 32.15 ± 0.88 |
| L3: + Device transfer | 24.33 ± 1.78 | 28.74 ± 1.00 | 33.03 ± 0.46 |
| L4: + Decompress & Apply (full) | 34.76 ± 5.95 | 46.41 ± 5.43 | 46.80 ± 5.96 |

### Median ITL (ms)

| Level | np=32 | np=64 | np=128 |
|-------|-------|-------|--------|
| L0: No structured output | 18.94 ± 0.25 | 19.34 ± 0.30 | 20.14 ± 0.21 |
| L1: Grammar FSM only | 19.32 ± 0.47 | 19.77 ± 0.32 | 20.35 ± 0.17 |
| L2: + Reorder + DP Gather | 20.11 ± 0.29 | 20.50 ± 0.34 | 21.26 ± 0.23 |
| L3: + Device transfer | 20.97 ± 0.42 | 20.91 ± 0.17 | 21.59 ± 0.27 |
| L4: + Decompress & Apply (full) | 24.26 ± 2.04 | 21.97 ± 0.38 | 21.34 ± 0.22 |

### P99 ITL (ms)

| Level | np=32 | np=64 | np=128 |
|-------|-------|-------|--------|
| L0: No structured output | 26.95 ± 6.34 | 174.13 ± 79.47 | 383.99 ± 0.75 |
| L1: Grammar FSM only | 26.44 ± 3.67 | 210.62 ± 0.72 | 318.89 ± 91.00 |
| L2: + Reorder + DP Gather | 25.50 ± 2.92 | 175.14 ± 79.76 | 384.18 ± 1.23 |
| L3: + Device transfer | 31.61 ± 4.97 | 179.25 ± 81.01 | 423.05 ± 77.30 |
| L4: + Decompress & Apply (full) | 67.75 ± 40.55 | 793.74 ± 476.13 | 1020.25 ± 363.76 |

## TPOT (Time Per Output Token)

### Mean TPOT (ms)

| Level | np=32 | np=64 | np=128 |
|-------|-------|-------|--------|
| L0: No structured output | 20.96 ± 2.20 | 28.52 ± 2.06 | 32.97 ± 1.66 |
| L1: Grammar FSM only | 23.79 ± 1.88 | 30.14 ± 1.57 | 32.53 ± 1.33 |
| L2: + Reorder + DP Gather | 24.20 ± 2.06 | 28.76 ± 1.36 | 32.94 ± 1.25 |
| L3: + Device transfer | 24.74 ± 2.08 | 30.40 ± 1.48 | 33.59 ± 0.70 |
| L4: + Decompress & Apply (full) | 35.11 ± 6.01 | 46.73 ± 5.46 | 47.11 ± 5.98 |

### Median TPOT (ms)

| Level | np=32 | np=64 | np=128 |
|-------|-------|-------|--------|
| L0: No structured output | 21.35 ± 2.81 | 26.70 ± 2.06 | 33.82 ± 2.21 |
| L1: Grammar FSM only | 19.97 ± 0.55 | 29.30 ± 2.63 | 32.49 ± 2.08 |
| L2: + Reorder + DP Gather | 20.44 ± 0.14 | 28.14 ± 2.25 | 32.86 ± 1.20 |
| L3: + Device transfer | 21.64 ± 0.85 | 28.57 ± 2.30 | 33.72 ± 1.26 |
| L4: + Decompress & Apply (full) | 26.52 ± 3.50 | 44.55 ± 4.58 | 48.23 ± 9.00 |

### P99 TPOT (ms)

| Level | np=32 | np=64 | np=128 |
|-------|-------|-------|--------|
| L0: No structured output | 22.87 ± 4.33 | 42.77 ± 4.38 | 46.57 ± 2.09 |
| L1: Grammar FSM only | 41.77 ± 1.68 | 44.46 ± 3.10 | 46.65 ± 1.82 |
| L2: + Reorder + DP Gather | 44.38 ± 4.07 | 47.04 ± 2.56 | 47.30 ± 4.97 |
| L3: + Device transfer | 45.61 ± 3.44 | 50.12 ± 6.98 | 47.18 ± 2.20 |
| L4: + Decompress & Apply (full) | 63.72 ± 6.83 | 80.13 ± 10.42 | 77.74 ± 19.33 |

## Per-Step Cost (Delta from Previous Level)

Each row shows the additional latency introduced by that step.

### Median ITL (ms) -- delta per step

| Step | np=32 | np=64 | np=128 |
|------|-------|-------|--------|
| L1: Grammar FSM only | +0.37 | +0.43 | +0.21 |
| L2: + Reorder + DP Gather | +0.79 | +0.73 | +0.91 |
| L3: + Device transfer | +0.87 | +0.41 | +0.34 |
| L4: + Decompress & Apply (full) | +3.28 | +1.06 | -0.25 |
| **Total overhead** | **+5.31** | **+2.63** | **+1.20** |

### Mean ITL (ms) -- delta per step

| Step | np=32 | np=64 | np=128 |
|------|-------|-------|--------|
| L1: Grammar FSM only | +2.05 | +1.88 | -0.65 |
| L2: + Reorder + DP Gather | +0.65 | -1.24 | +0.33 |
| L3: + Device transfer | +1.18 | +1.33 | +0.88 |
| L4: + Decompress & Apply (full) | +10.43 | +17.67 | +13.77 |
| **Total overhead** | **+14.31** | **+19.64** | **+14.33** |

### P99 ITL (ms) -- delta per step

| Step | np=32 | np=64 | np=128 |
|------|-------|-------|--------|
| L1: Grammar FSM only | -0.51 | +36.49 | -65.10 |
| L2: + Reorder + DP Gather | -0.94 | -35.48 | +65.30 |
| L3: + Device transfer | +6.11 | +4.11 | +38.87 |
| L4: + Decompress & Apply (full) | +36.14 | +614.49 | +597.20 |
| **Total overhead** | **+40.79** | **+619.61** | **+636.26** |

### Median TPOT (ms) -- delta per step

| Step | np=32 | np=64 | np=128 |
|------|-------|-------|--------|
| L1: Grammar FSM only | -1.38 | +2.60 | -1.33 |
| L2: + Reorder + DP Gather | +0.47 | -1.16 | +0.37 |
| L3: + Device transfer | +1.20 | +0.43 | +0.85 |
| L4: + Decompress & Apply (full) | +4.88 | +15.98 | +14.52 |
| **Total overhead** | **+5.17** | **+17.84** | **+14.41** |

### Mean TPOT (ms) -- delta per step

| Step | np=32 | np=64 | np=128 |
|------|-------|-------|--------|
| L1: Grammar FSM only | +2.83 | +1.62 | -0.44 |
| L2: + Reorder + DP Gather | +0.41 | -1.38 | +0.41 |
| L3: + Device transfer | +0.54 | +1.64 | +0.65 |
| L4: + Decompress & Apply (full) | +10.36 | +16.32 | +13.52 |
| **Total overhead** | **+14.15** | **+18.20** | **+14.14** |

### P99 TPOT (ms) -- delta per step

| Step | np=32 | np=64 | np=128 |
|------|-------|-------|--------|
| L1: Grammar FSM only | +18.90 | +1.69 | +0.08 |
| L2: + Reorder + DP Gather | +2.61 | +2.58 | +0.65 |
| L3: + Device transfer | +1.23 | +3.08 | -0.12 |
| L4: + Decompress & Apply (full) | +18.11 | +30.01 | +30.56 |
| **Total overhead** | **+40.84** | **+37.36** | **+31.17** |

## Throughput

### Output Token Throughput (tok/s)

| Level | np=32 | np=64 | np=128 |
|-------|-------|-------|--------|
| L0: No structured output | 468.64 ± 113.22 | 565.98 ± 47.62 | 633.63 ± 56.40 |
| L1: Grammar FSM only | 455.55 ± 91.00 | 536.55 ± 34.67 | 609.59 ± 37.53 |
| L2: + Reorder + DP Gather | 365.18 ± 46.37 | 560.77 ± 38.54 | 623.87 ± 23.67 |
| L3: + Device transfer | 337.98 ± 58.12 | 513.39 ± 63.34 | 600.80 ± 19.04 |
| L4: + Decompress & Apply (full) | 410.32 ± 18.98 | 366.67 ± 15.53 | 413.67 ± 30.09 |

## Notes

- All levels use the same benchmark (`benchmark_serving_structured_output.py`
  with `--dataset json-unique --output-len 1000`). L0 adds `--no-structured-output`
  to use identical prompts without grammar constraints, making L0-L4 directly comparable.
- Each level ran with a fresh server restart and device reset.
- The grammar FSM cost (L0->L1) appears to scale with np because higher np
  means more concurrent requests causing prefill interruptions, not because
  per-step FSM work increases (batch size per DP rank is always 8).
- Decompress+Apply (L3->L4) shows large cost in mean/P99 metrics but
  more modest cost in median ITL, indicating tail latency spikes when
  device-side bitmask ops coincide with prefill scheduling.
