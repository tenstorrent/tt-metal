# Fused AllGather + MatMul (AG+MM) for LLaMA 70B Galaxy — W2/FF2 Path

## Overview

Integrates `ttnn.experimental.all_gather_minimal_matmul_async` into the LLaMA 70B Galaxy prefill MLP W2 path, replacing separate AllGather + MinimalMatmul ops with a single fused op.

Based on PR #36711 (jonathansu/agmm-combined).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_FUSED_AG_MM_PREFILL` | `0` | Enable fused AG+MM for W2 prefill path |
| `USE_PADDED_W2` | `0` | Pad W2 weight N dim (2048→2240) to enable 7x8 grid (56 cores) |
| `FUSED_AG_MM_GRID_X` | `4` (or `7` if padded) | Grid X (columns) for fused op |
| `FUSED_AG_MM_GRID_Y` | `8` | Grid Y (rows) for fused op |
| `FUSED_AG_MM_K_BLOCK` | `8` | K block size (8 recommended) |

## Run Commands

### Baseline (no fusion)
```bash
./scripts/run_profiler_sweep.sh --run-name baseline_8k --prompt-lengths 8k
```

### Fused AG+MM (4x8 grid, 32 cores)
```bash
USE_FUSED_AG_MM_PREFILL=1 ./scripts/run_profiler_sweep.sh --run-name fused_ag_mm_8k --prompt-lengths 8k
```

### Fused AG+MM with Padded W2 (7x8 grid, 56 cores)
```bash
USE_PADDED_W2=1 USE_FUSED_AG_MM_PREFILL=1 ./scripts/run_profiler_sweep.sh --run-name fused_ag_mm_padded_8k --prompt-lengths 8k
```

### Other sequence lengths
```bash
USE_FUSED_AG_MM_PREFILL=1 ./scripts/run_profiler_sweep.sh --run-name fused_ag_mm_16k --prompt-lengths 16k
USE_FUSED_AG_MM_PREFILL=1 ./scripts/run_profiler_sweep.sh --run-name fused_ag_mm_64k --prompt-lengths 64k
USE_FUSED_AG_MM_PREFILL=1 ./scripts/run_profiler_sweep.sh --run-name fused_ag_mm_128k --prompt-lengths 128k
```

## Grid Constraints (Harvested Galaxy, 7 columns)

W2 weight per device: K=3584, N=2048. N_tiles = 64.

**Without padding (N=2048, 64 tiles):**
- grid_x must divide 64 AND be ≤ 7 → only {1, 2, 4}
- Best: 4x8 = 32 cores

**With padding (N=2240, 70 tiles):**
- grid_x=7: 70/7=10 per core ✓
- Best: 7x8 = 56 cores
- Uses N_block_size=10 instead of 8

## Results (4x8 grid, Block 8/8/8, HiFi2)

| ISL | Baseline AG (us) | Baseline MM (us) | Baseline Total | Fused (us) | Fused Cores | Delta |
|-----|-----------------|-----------------|---------------|------------|-------------|-------|
| 8k  | 524 | 1318 | 1842 | 2208 | 34 | +20% |
| 16k | 1013 | 2611 | 3624 | 4376 | 34 | +21% |
| 32k | 2026 | 4999 | 7025 | 8710 | 34 | +24% |
| 64k | 4023 | 9854 | 13877 | 17397 | 34 | +25% |
| 128k | 8011 | 19622 | 27632 | 34444 | 34 | +25% |

## Files Modified

- `models/demos/llama3_70b_galaxy/tt/model_config.py` — fused config, env vars
- `models/demos/llama3_70b_galaxy/tt/llama_mlp.py` — fused op integration, padded weight
- `models/demos/llama3_70b_galaxy/tt/llama_ccl.py` — semaphores, persistent buffers
- `models/demos/llama3_70b_galaxy/tt/generator.py` — UInt32 sampling fix
- `scripts/run_profiler_sweep.sh` — profiler sweep runner
- `scripts/parse_profiler_report.py` — profiler CSV parser
