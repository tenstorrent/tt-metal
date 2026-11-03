# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_57_31
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

### num_links=2, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_02_09_57_43/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.5       | 55.160   | 0.055  | 70.0       | 73.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5422.4     | 4016.593 | 4.017  | 5353.0     | 5420.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5530.1     | 4096.346 | 4.096  | 5338.0     | 5475.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5355.3     | 3966.914 | 3.967  | 3840.0     | 5393.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5392.6     | 3994.519 | 3.995  | 2949.0     | 5465.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5414.2     | 4010.494 | 4.010  | 5315.0     | 5399.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5572.0     | 4127.407 | 4.127  | 5390.0     | 5469.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5387.5     | 3990.741 | 3.991  | 4854.0     | 5408.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5433.3     | 4024.667 | 4.025  | 4078.0     | 5475.5        |

Final op time: 4.127 us (5572.0 cycles), device=5, bucket=NCRISC transport

