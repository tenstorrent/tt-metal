# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_16_46
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

### num_links=1, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_02_09_16_57/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 73.3       | 54.272   | 0.054  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5572.8     | 4128.025 | 4.128  | 5454.0     | 5593.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5697.6     | 4220.420 | 4.220  | 5425.0     | 5668.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5535.2     | 4100.173 | 4.100  | 3823.0     | 5580.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5543.7     | 4106.420 | 4.106  | 2467.0     | 5629.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5572.2     | 4127.580 | 4.128  | 5325.0     | 5566.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5706.7     | 4227.160 | 4.227  | 5494.0     | 5655.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5547.0     | 4108.889 | 4.109  | 4018.0     | 5570.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5567.0     | 4123.704 | 4.124  | 3171.0     | 5623.0        |

Final op time: 4.227 us (5706.7 cycles), device=5, bucket=NCRISC transport

