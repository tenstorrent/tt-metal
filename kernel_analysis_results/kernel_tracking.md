# Kernel Analysis Tracking

## Summary
- Total files with `noc_async_write_barrier`: 372
- Total files with fabric write APIs: ~24

## File Categories

### 1. CCL/Fabric Operations (experimental/ccl + ccl)
Files in `ttnn/cpp/ttnn/operations/ccl/` and `ttnn/cpp/ttnn/operations/experimental/ccl/`

### 2. Data Movement Operations
Files in `ttnn/cpp/ttnn/operations/data_movement/`

### 3. Moreh Operations
Files in `ttnn/cpp/ttnn/operations/moreh/`

### 4. Transformer/Matmul Operations
Files in `ttnn/cpp/ttnn/operations/transformer/`, `ttnn/cpp/ttnn/operations/matmul/`, `ttnn/cpp/ttnn/operations/experimental/transformer/`

### 5. Other Operations
Remaining files in `ttnn/cpp/ttnn/operations/` (reduction, normalization, eltwise, conv, etc.)

### 6. Tests and Examples
Files in `tests/` and `tt_metal/programming_examples/`

## Analysis Status
- [ ] CCL/Fabric analysis - Pending
- [ ] Data Movement analysis - Pending
- [ ] Moreh analysis - Pending
- [ ] Transformer/Matmul analysis - Pending
- [ ] Other Operations analysis - Pending
