# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_50_25
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

Report: generated/profiler/reports/2026_11_02_09_50_38/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.5       | 56.642   | 0.057  | 70.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9352.7     | 6927.951 | 6.928  | 9264.0     | 9355.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9414.7     | 6973.877 | 6.974  | 9247.0     | 9407.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9250.3     | 6852.074 | 6.852  | 6137.0     | 9342.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9232.2     | 6838.642 | 6.839  | 3990.0     | 9401.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9321.1     | 6904.519 | 6.905  | 8725.0     | 9330.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9412.1     | 6971.926 | 6.972  | 9294.0     | 9407.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9265.8     | 6863.556 | 6.864  | 6663.0     | 9336.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9272.5     | 6868.519 | 6.869  | 5169.0     | 9413.0        |

Final op time: 6.974 us (9414.7 cycles), device=1, bucket=NCRISC transport

