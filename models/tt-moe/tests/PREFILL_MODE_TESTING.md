# Prefill Mode Testing for DeepSeek MoE Block

## Overview
This document describes the parameterization added to `test_deepseek_moe_block.py` to support testing both decode and prefill modes for the DeepSeek MoE infrastructure.

## Changes Made

### 1. Test Parameterization
Added `@pytest.mark.parametrize` decorator to `test_deepseek_moe_against_reference`:
- **Decode mode**: `seq_len=1`, `batch_size=32` per row (128 total for 4 rows)
- **Prefill mode**: `seq_len=128`, `batch_size=1` per row (4 total for 4 rows)

### 2. Batch Size Adjustment
Different batch sizes are used for each mode following the reference implementation pattern:
- **Decode**: Higher batch size (32) with minimal sequence length (1)
- **Prefill**: Lower batch size (1) with full sequence length (128)
- This balances memory usage and computational efficiency

### 3. Input/Output Handling
Updated tensor reshaping logic to handle both modes correctly:
- **Decode**: Input shape `[batch, 1, hidden]` → `[1, 1, batch, hidden]` for TTNN
- **Prefill**: Input shape `[batch, seq_len, hidden]` → `[1, batch, seq_len, hidden]` for TTNN

### 4. Test Identification
Added clear test IDs for easier identification:
- `mode_decode_seq_1`
- `mode_prefill_seq_128`

## Running the Tests

### Run specific mode:
```bash
# Decode mode only
pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference[mode_decode_seq_1] -xvs

# Prefill mode only
pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference[mode_prefill_seq_128] -xvs
```

### Run both modes:
```bash
pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference -xvs
```

### Use convenience script:
```bash
# Run decode mode
./run_deepseek_tests.sh decode

# Run prefill mode
./run_deepseek_tests.sh prefill

# Run both modes
./run_deepseek_tests.sh both
```

## Technical Details

### Memory Considerations
- Prefill mode uses smaller batch sizes to accommodate longer sequences in memory
- The infrastructure includes automatic chunking support for very large prefill sequences
- Chunk size is configurable via `prefill_chunk_size` in the JSON config (default: 16384)

### PCC Threshold
Both modes maintain the same PCC (Pearson Correlation Coefficient) threshold of 0.98 for accuracy validation.

### Configuration
The existing `deepseek_v3.json` configuration already supports both modes with appropriate chunking parameters:
- `moe_chunk_size`: 4096 (for general MoE operations)
- `prefill_chunk_size`: 16384 (for large prefill sequences)

## Compatibility
The changes maintain full backward compatibility:
- Existing decode mode test continues to work as before
- All test fixtures and utilities remain unchanged
- The reference model comparison logic works for both modes

## Future Enhancements
Potential improvements for future iterations:
1. Add more sequence length variations (e.g., 256, 512, 1024)
2. Test different batch size combinations
3. Add performance benchmarking for each mode
4. Test chunking behavior with very long sequences
