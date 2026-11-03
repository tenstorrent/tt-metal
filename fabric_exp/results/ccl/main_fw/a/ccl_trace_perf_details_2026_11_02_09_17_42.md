# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_17_42
CCLs: broadcast

## broadcast

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_broadcast.py::test_ccl_broadcast
Configured benchmark shape: [1, 7168]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = slowest device's slowest top-level bucket, using per-device averages across iterations only.

Cross-device bucket averages are omitted for this CCL.

### num_links=2, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_02_09_17_53/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.5       | 56.691   | 0.057  | 70.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9447.7     | 6998.296 | 6.998  | 9245.0     | 9473.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9487.0     | 7027.407 | 7.027  | 8850.0     | 9495.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9336.6     | 6916.025 | 6.916  | 6035.0     | 9435.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9335.6     | 6915.235 | 6.915  | 4263.0     | 9501.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9395.5     | 6959.654 | 6.960  | 8461.0     | 9460.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9503.0     | 7039.259 | 7.039  | 9197.0     | 9546.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9359.4     | 6932.889 | 6.933  | 6518.0     | 9446.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9363.1     | 6935.605 | 6.936  | 5024.0     | 9515.5        |

Final op time: 7.039 us (9503.0 cycles), device=5, bucket=NCRISC transport

