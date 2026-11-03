# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_16_35
CCLs: broadcast

## broadcast

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_broadcast.py::test_ccl_broadcast
Configured benchmark shape: [1, 7168]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = slowest device's slowest top-level bucket, using per-device averages across iterations only.

Cross-device bucket averages are omitted for this CCL.

### num_links=1, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_02_09_16_46/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 73.2       | 54.247   | 0.054  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6773.3     | 5017.259 | 5.017  | 6684.0     | 6797.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6877.3     | 5094.296 | 5.094  | 6721.0     | 6861.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6730.2     | 4985.309 | 4.985  | 4607.0     | 6785.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6721.4     | 4978.790 | 4.979  | 3035.0     | 6828.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6765.0     | 5011.136 | 5.011  | 6536.0     | 6785.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6897.7     | 5109.383 | 5.109  | 6708.0     | 6848.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6742.3     | 4994.272 | 4.994  | 4798.0     | 6804.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6743.6     | 4995.235 | 4.995  | 3949.0     | 6813.5        |

Final op time: 5.109 us (6897.7 cycles), device=5, bucket=NCRISC transport

