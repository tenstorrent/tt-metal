# TG MoE Test Suite for Quad Galaxy Validation

This directory contains tests that run on **single galaxy (TG, 32 devices, 4x8 mesh)** to validate that changes won't break the quad galaxy (128 devices, 16x8 mesh) MoE implementation.

## Test Philosophy

**Goal**: If a change breaks the quad demo, it should also break these TG tests.

**Key Principle**: Tests maintain the same **per-device workload** as quad:
- Quad: 256 experts / 128 devices = 2 experts/device
- TG:   64 experts / 32 devices = 2 experts/device

## Test Types

### 1. E2E Test (`test_optimized_moe_decode_block_tg.py`)
Full optimized MoE decode block on 4x8 mesh:
- Dispatch → Compute → Combine → Tilize → Scale → Reduce

**Note**: Uses non-optimized reduce operations (`fast_reduce_nc` + `reduce_scatter`) since optimized versions are hardcoded for 8-device setups (cluster_axis=1).

### 2. Individual Op Tests (`individual_ops/`)
Tests for each operation in isolation:
- `test_dispatch_tg.py` - all_to_all_dispatch_metadata
- `test_compute_tg.py` - moe_compute
- `test_combine_tg.py` - selective_reduce_combine
- `test_reduce_tg.py` - non-optimized reduce operations

These catch input/kernel errors at the op level before running expensive E2E tests.

### 3. Existing 1x8/1x16 Tests (moved here)
- `test_all_to_all_dispatch_metadata_6U.py`
- `test_moe_compute_6U.py`
- `test_selective_combine_6U.py`

## Scaled Parameters

| Parameter | Quad (16x8) | TG (4x8) | Ratio |
|-----------|-------------|----------|-------|
| Devices | 128 | 32 | 4:1 |
| Experts | 256 | 64 | 4:1 |
| Experts/device | 2 | 2 | 1:1 ✓ |
| Batch (decode) | 512 | 128 | 4:1 |
| Batches/device | 32 | 32 | 1:1 ✓ |
| Selected experts (k) | 8 | 8 | 1:1 ✓ |
| Hidden size | 7168 | 7168 | 1:1 ✓ |
| Matmul N | 2048 | 2048 | 1:1 ✓ |

## Running Tests

```bash
# Set mesh to TG (4x8)
export MESH_DEVICE=TG

# For E2E test, also need ring fabric mode
export USE_TORUS_MODE=1

# Run E2E test
pytest models/demos/deepseek_v3/tests/tg_moe_tests/test_optimized_moe_decode_block_tg.py -v

# Run individual op tests (don't need USE_TORUS_MODE)
pytest models/demos/deepseek_v3/tests/tg_moe_tests/individual_ops/ -v

# Run specific test
pytest models/demos/deepseek_v3/tests/tg_moe_tests/individual_ops/test_dispatch_tg.py -v
```

### Quick Test Commands

**Dispatch test only:**
```bash
export MESH_DEVICE=TG
pytest models/demos/deepseek_v3/tests/tg_moe_tests/individual_ops/test_dispatch_tg.py::test_correctness -v
```

**E2E test:**
```bash
export MESH_DEVICE=TG
export USE_TORUS_MODE=1
pytest models/demos/deepseek_v3/tests/tg_moe_tests/test_optimized_moe_decode_block_tg.py -v
```

## Validation Strategy

While working on this project, you should:
1. Make random breaking changes to ops (kernels or program factories)
2. Assert that one of the TG tests fails
3. This validates the test suite catches real breaking changes

## Known Limitations

1. **Optimized reduce-scatter**: Not compatible with 4x8 mesh (requires 8 devices on cluster_axis opposite of dispatch). Tests use standard reduce operations.

2. **Fast reduce optimizations**: `deepseek_moe_fast_reduce_nc` with specific shard specs for 8-device setups. Tests use standard version.

3. **Memory configs**: Some optimized memory configs are quad-specific. TG tests use compatible configs.

## Contributing

When adding new optimized ops to the quad path:
1. Add corresponding TG test here (use non-optimized version if needed)
2. Ensure test fails when op is broken
3. Run full test suite before merging quad changes
