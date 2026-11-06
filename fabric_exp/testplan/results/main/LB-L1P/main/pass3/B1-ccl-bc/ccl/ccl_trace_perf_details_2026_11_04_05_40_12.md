# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_40_12
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

### num_links=1, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_04_05_40_22/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.6       | 56.765   | 0.057  | 75.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6824.2     | 5054.938 | 5.055  | 6704.0     | 6827.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6954.0     | 5151.136 | 5.151  | 6705.0     | 6905.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6784.9     | 5025.852 | 5.026  | 4604.0     | 6880.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6770.0     | 5014.790 | 5.015  | 2875.0     | 6931.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6814.2     | 5047.580 | 5.048  | 6149.0     | 6831.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6951.9     | 5149.531 | 5.150  | 6726.0     | 6876.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6779.1     | 5021.580 | 5.022  | 4452.0     | 6829.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6783.3     | 5024.667 | 5.025  | 3402.0     | 6875.0        |

Final op time: 5.151 us (6954.0 cycles), device=1, bucket=NCRISC transport

