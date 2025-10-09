# Dump Remaining Buffers Feature

## Overview
The allocation server now supports dumping all remaining allocated buffers to help identify potential memory leaks.

## Usage

### 1. Start the allocation server
```bash
./allocation_server_poc
```

### 2. Run your test/application
```bash
# Example: Run LLaMA inference
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"
```

### 3. After the test completes, dump remaining buffers
```bash
python3 dump_remaining_buffers.py
```

### 4. Check the server output
The server will print a detailed report showing:
- Buffers grouped by device
- Buffers grouped by type (DRAM, L1, L1_SMALL, TRACE)
- Individual buffer details (address, size, PID, ref_count)

## Example Output

```
╔══════════════════════════════════════════════════════════════╗
║           REMAINING ALLOCATED BUFFERS                       ║
╚══════════════════════════════════════════════════════════════╝

Device 0:
  DRAM: 2 buffers, 0.036864 MB total
    - Buffer 0x40000000: 12.0 KB (PID 12345, ref_count=1)
    - Buffer 0x40003000: 24.0 KB (PID 12345, ref_count=1)
  L1: 5 buffers, 5.79 MB total

Device 1:
  DRAM: 2 buffers, 0.036864 MB total
  L1: 3 buffers, 6.70 MB total

...

Total remaining buffers: 42
```

## Interpreting Results

### Expected Remaining Buffers
- **Cached programs**: Kernel binaries (DRAM, ~12-24KB per program)
- **Model state**: KV cache, persistent weights (L1, several MB)
- **Infrastructure**: System buffers, command queues

### Potential Leaks
- Buffers with high `ref_count` that should have been freed
- Unexpected buffer types or sizes
- Buffers that accumulate across multiple runs

### To Clear Cached Programs
```python
mesh_device.disable_and_clear_program_cache()
```

## Notes
- Buffers allocated before tracking started will NOT appear (they were never tracked)
- The dump shows only buffers that were allocated AFTER the server started tracking
- Use this feature at the end of your test to verify proper cleanup
