# Chunk Size Performance Sweep

## Overview

`sweep_chunk_sizes.py` optimizes `q_chunk_size` and `k_chunk_size` parameters for `test_mla_sdpa_bh_galaxy` by sweeping through all valid combinations and measuring performance using the tracy profiler.

## Usage

### Prerequisites

1. Run from the repository root directory (e.g., `/tt-metal` or `/home/ubuntu/devs/tt-metal`)
2. Python environment must be activated
3. Tracy profiler must be available (`python3 -m tracy`)

### Running the Sweep

```bash
# From the repository root (e.g., /tt-metal or /home/ubuntu/devs/tt-metal)
python3 scripts/sweep_chunk_sizes.py
```

The script will:
1. Create a backup of the test file
2. Test all valid chunk size combinations (32, 64, 128, 256, 512, 1024, 2048)
3. Stop testing larger values when OOM errors occur
4. Parse tracy profiler CSV outputs for each test
5. Save results and identify the optimal configuration
6. Restore the original test file

## Valid Chunk Sizes

Both `q_chunk_size` and `k_chunk_size` must:
- Start at 32 and increment by 32
- Divide 2048 evenly
- Valid values: 32, 64, 128, 256, 512, 1024, 2048

## Output Files

- `sweep_results_YYYY_MM_DD_HH_MM_SS.csv` - All test results
- `sweep_log_YYYY_MM_DD_HH_MM_SS.txt` - Detailed execution log

### Results CSV Format

```
q_chunk_size,k_chunk_size,max_time_ns,min_time_ns,status
32,32,2886000000,2664000000,SUCCESS
32,64,2456000000,2301000000,SUCCESS
...
256,512,,,FAILED/OOM
```

## Performance Metrics

- **MAX time**: Slowest device kernel duration (bottleneck) across all 32 chips
- **MIN time**: Fastest device kernel duration
- **Goal**: Minimize the MAX time (the limiting factor)

## Features

### Automatic Backup & Restore
- Test file is automatically backed up before modifications
- Original file is restored on completion, error, or interrupt (Ctrl+C)

### Progressive Results
- Results are saved after each test
- Safe to interrupt (Ctrl+C) without losing progress

### OOM Detection
- Automatically detects out-of-memory errors
- Stops testing larger chunk sizes when OOM occurs
- Marks failed tests in results

### Detailed Logging
- Complete test output logged to file
- Console shows progress and key metrics
- Timestamps for all events

## Example Output

```
============================================================
Starting chunk size sweep...
Valid chunk sizes: [32, 64, 128, 256, 512, 1024, 2048]
Test file: models/tt_dit/tests/unit/test_ring_joint_mla.py
Test name: test_mla_sdpa_bh_galaxy
============================================================

--- Testing q_chunk_size = 32 ---

Testing q_chunk_size=32, k_chunk_size=32
  ✓ MAX: 2.886s, MIN: 2.664s

Testing q_chunk_size=32, k_chunk_size=64
  ✓ MAX: 2.456s, MIN: 2.301s

...

Testing q_chunk_size=256, k_chunk_size=512
  FAILED - stopping further increases for k_chunk_size=512

Results saved to: sweep_results_2026_03_06_16_30_00.csv

============================================================
=== OPTIMAL CONFIGURATION ===
============================================================
q_chunk_size: 128
k_chunk_size: 256
MAX time: 1.234s
MIN time: 1.156s
Speedup vs baseline (32,32): 2.34x
============================================================

=== ALL CONFIGURATIONS (SORTED BY PERFORMANCE) ===
1. q=128, k=256: MAX=1.234s, MIN=1.156s
2. q=128, k=128: MAX=1.345s, MIN=1.267s
3. q=64, k=256: MAX=1.456s, MIN=1.378s
4. q=256, k=128: MAX=1.567s, MIN=1.489s
5. q=64, k=128: MAX=1.678s, MIN=1.590s
6. q=32, k=256: MAX=1.789s, MIN=1.701s
...

Total sweep time: 2:34:56
```

## Implementation Details

### Tracy Profiler Integration
- Runs: `python -m tracy -r -m pytest models/tt_dit/tests/unit/test_ring_joint_mla.py::test_mla_sdpa_bh_galaxy`
- Parses: `/tt-metal/generated/profiler/reports/{timestamp}/ops_perf_results_{timestamp}.csv`
- Extracts: `DEVICE KERNEL DURATION [ns]` for all 32 devices

### Test File Modification
- Modifies line 615 of `test_ring_joint_mla.py`
- Updates parameter tuple: `(1, 128, 1, 128, 128*1024, 576, 576, 128, True, q_chunk, k_chunk)`
- Preserves all other parameters

### OOM Handling
- Checks pytest exit code for failures
- Searches stderr for "out of memory" or "oom"
- Stops increasing k_chunk_size when OOM detected
- Continues with next q_chunk_size value

## Troubleshooting

### "Test file not found" error
Run the script from the `/tt-metal` directory.

### No CSV file generated
Check the log file for test failures. The test might be failing before profiling completes.

### OOM on small chunk sizes
Verify the test setup is correct and sufficient memory is available.

### Script hangs
Default timeout is 30 minutes per test. Very large chunk sizes may exceed this.

## Notes

- Each test takes several minutes to complete
- Full sweep can take several hours depending on hardware
- Script is safe to interrupt - results are saved progressively
- The test file is always restored, even on errors or interrupts
