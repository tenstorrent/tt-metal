# Fused MatMul + ReduceScatter (MM+RS) Integration

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_FUSED_MM_RS_PREFILL` | `0` | Enable fused MM+RS for W1/FF1 in prefill |
| `CONSTRAIN_FF1_CORES` | `0` | Force non-fused W1 MM to use same 4x8 grid (apples-to-apples comparison) |
| `FUSED_GRID_X` | `4` | MM core grid columns (only 4 works on harvested Galaxy due to divisibility constraint) |
| `FUSED_GRID_Y` | `8` | MM core grid rows |
| `FUSED_CHUNK_WIDTH` | `2` | chunk_width_in_mm_blocks for RS signaling granularity |
| `FUSED_NUM_WORKERS` | `3` | num_workers_per_link for RS |

## Run Commands

### Baseline (no fusion)
```bash
./scripts/run_profiler_sweep.sh --run-name baseline_8k --prompt-lengths 8k
```

### Fused MM+RS
```bash
USE_FUSED_MM_RS_PREFILL=1 ./scripts/run_profiler_sweep.sh --run-name fused_rs_mm_8k --prompt-lengths 8k
```

### Constrained separate (apples-to-apples: same 4x8 grid, but MM and RS run separately)
```bash
CONSTRAIN_FF1_CORES=1 ./scripts/run_profiler_sweep.sh --run-name constrained_4x8_8k --prompt-lengths 8k
```

### Fused at 128k
```bash
USE_FUSED_MM_RS_PREFILL=1 FUSED_GRID_X=4 FUSED_GRID_Y=8 ./scripts/run_profiler_sweep.sh --run-name fused_128k --prompt-lengths 128k
```

### Custom grid (will crash if divisibility constraint fails)
```bash
USE_FUSED_MM_RS_PREFILL=1 FUSED_GRID_X=7 FUSED_GRID_Y=8 ./scripts/run_profiler_sweep.sh --run-name test_7x8 --prompt-lengths 8k
```

## Results (8k seq_len, W1/FF1)

| Run | Op | Time (us) | Cores |
|-----|-----|-----------|-------|
| Baseline | W1 MatMul | 744 | 63 |
| Baseline | W1 ReduceScatter | 687 | 40 |
| Baseline | **Total** | **1,431** | — |
| Constrained 4x8 | W1 MatMul | 1,309 | 32 |
| Constrained 4x8 | W1 ReduceScatter | 690 | 40 |
| Constrained 4x8 | **Total** | **1,999** | — |
| Fused | MM+RS combined | **2,960** | 40 |

## Key Constraint: `slice_Wt % mm_N_block_wt == 0`

- `slice_Wt = N_tiles / (ring_size / 2) = 112 / 4 = 28`
- `mm_N_block_wt = round_up(N_tiles, grid_x) / grid_x`
- On harvested Galaxy (7 columns), only `grid_x=4` satisfies this (28 % 28 = 0)
- `grid_x=7` fails: 28 % 16 = 12
- This limits MM to 32 cores (4x8) vs baseline's 56-63 cores

## Standalone Pytest (test configurations without running full model)

```bash
# Run all configs
pytest tests/test_fused_mm_rs_galaxy.py -svv --timeout=300

# Run a specific config
pytest tests/test_fused_mm_rs_galaxy.py -svv -k "ff1_8k_8x8_fused" --timeout=300

# Run only separate (non-fused) baseline
pytest tests/test_fused_mm_rs_galaxy.py -svv -k "separate" --timeout=300
```

Edit `TEST_CONFIGS` in `tests/test_fused_mm_rs_galaxy.py` to experiment with different
grid sizes, block sizes, chunk widths, and num_workers_per_link.

**Important constraints when adding configs:**
- `slice_Wt % mm_N_block_wt == 0` (slice_Wt = N_tiles/ring_size, mm_N_block_wt = N_tiles/grid_x)
- `slice_Ht % grid_y == 0` (slice_Ht = M_tiles)
- `N_per_core % N_block_size == 0` (N_per_core = N_tiles/grid_x, N_block_size = mm_block_n/32)
- Device must have enough columns for grid_x (7 on harvested, 8 on unharvested)

## Files Modified

- `models/demos/llama3_70b_galaxy/tt/model_config.py` — env var configs, fused MM+RS config
- `models/demos/llama3_70b_galaxy/tt/llama_mlp.py` — fused op call in forward_prefill
- `tests/test_fused_mm_rs_galaxy.py` — standalone pytest for testing fused op configurations
