# Matmul Auto-Configuration Infrastructure Proposal

## Overview

This proposal describes an infrastructure to automatically select the most optimal
matmul configuration for any given input shape and device. The system uses a
two-tier approach: a fast heuristic-based analytical model for cold-start
decisions, and a performance database for learned optimal configurations.

## Architecture

### 1. MatmulConfigHeuristic (Analytical Cost Model)

A rule-based engine that selects config type and parameters based on:

- **Shape properties**: M, K, N dimensions, aspect ratio (narrow vs wide)
- **Memory layout**: sharded (height/width/block) vs interleaved
- **Device topology**: core grid size, number of devices
- **Data type**: fp32 vs bf16

**Selection algorithm**:
1. If narrow shape (ratio > 8x or dim <= 32): use 1D systolic array config
2. If width-sharded: use 1D mcast_in0=true config
3. If height-sharded: use 1D mcast_in0=false config
4. If block-sharded: use 2D multicast config with transposed mcast
5. If multi-device: partition N across devices, then recurse
6. Default: 2D multicast with auto-computed block sizes

### 2. MatmulConfigDatabase (Learned Performance DB)

Persists benchmark results keyed by `(M, K, N, batch_size, num_devices)`:

- **Registration**: benchmark a config on real hardware, record avg runtime
- **Query**: exact match or closest compatible shape
- **Persistence**: pickle file + JSON export for inspection

### 3. MatmulAutoConfig API

```
ttnn.matmul_auto(a, b) -> Tensor  # Torch.matmul-like signature
```

## Integration with Existing Code

- Uses existing `ttnn.MatmulProgramConfig` variants
- Calls `ttnn.matmul()` under the hood with the chosen config
- Exposed as `ttnn.matmul_auto()` for easy migration
- Performance data stored in `.opencode/matmul_db.pkl`

## Updating When Underlying Code Changes

1. Run benchmark suite on representative shapes
2. Update heuristic thresholds if new kernel implementations change perf
3. Rebuild performance database with `MatmulAutoConfig.benchmark_and_register()`
4. The database automatically adjusts as new samples are collected

## Reusability

The same infrastructure can be extended to other ops:
- Conv2d auto-config (kernel size, stride sharding)
- LayerNorm auto-config (normalized shape, sharding)
- Embedding auto-config (vocab size, batch size)

## Tests

- Unit tests for heuristic selection logic
- Compatibility tests for shape matching
- Database persistence tests
- Validation that all config types are reachable
