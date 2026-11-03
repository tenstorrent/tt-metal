# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_23_36
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

### num_links=1, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_02_09_23_49/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.8       | 55.432   | 0.055  | 74.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5050.1     | 3740.815 | 3.741  | 4986.0     | 5039.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5166.7     | 3827.210 | 3.827  | 5056.0     | 5132.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5061.1     | 3748.963 | 3.749  | 4137.0     | 5095.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5102.5     | 3779.605 | 3.780  | 4065.0     | 5128.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5088.2     | 3769.037 | 3.769  | 4932.0     | 5066.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5219.8     | 3866.519 | 3.867  | 5038.0     | 5149.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5113.1     | 3787.481 | 3.787  | 4983.0     | 5098.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5137.5     | 3805.531 | 3.806  | 5064.0     | 5135.0        |

Final op time: 3.867 us (5219.8 cycles), device=5, bucket=NCRISC transport

