# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_57_17
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

Report: generated/profiler/reports/2026_11_02_09_57_30/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.5       | 56.691   | 0.057  | 70.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9357.0     | 6931.136 | 6.931  | 9266.0     | 9353.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9377.7     | 6946.469 | 6.946  | 8363.0     | 9401.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9265.1     | 6863.037 | 6.863  | 6578.0     | 9344.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9276.6     | 6871.531 | 6.872  | 5260.0     | 9404.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9314.0     | 6899.235 | 6.899  | 8571.0     | 9330.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9403.5     | 6965.580 | 6.966  | 9071.0     | 9400.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9284.9     | 6877.704 | 6.878  | 7104.0     | 9351.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9299.1     | 6888.222 | 6.888  | 5969.0     | 9379.5        |

Final op time: 6.966 us (9403.5 cycles), device=5, bucket=NCRISC transport

