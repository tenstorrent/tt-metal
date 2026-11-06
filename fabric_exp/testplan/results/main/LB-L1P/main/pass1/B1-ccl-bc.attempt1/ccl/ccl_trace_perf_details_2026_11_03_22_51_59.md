# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_51_59
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

Report: generated/profiler/reports/2026_11_03_22_52_10/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.6       | 56.741   | 0.057  | 75.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6824.4     | 5055.136 | 5.055  | 6680.0     | 6832.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6928.6     | 5132.272 | 5.132  | 6674.0     | 6868.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6769.2     | 5014.198 | 5.014  | 4554.0     | 6865.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6760.8     | 5008.025 | 5.008  | 2889.0     | 6902.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6815.6     | 5048.568 | 5.049  | 6359.0     | 6825.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6942.6     | 5142.667 | 5.143  | 6687.0     | 6921.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6779.1     | 5021.580 | 5.022  | 4939.0     | 6840.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6770.9     | 5015.481 | 5.015  | 3332.0     | 6886.0        |

Final op time: 5.143 us (6942.6 cycles), device=5, bucket=NCRISC transport

