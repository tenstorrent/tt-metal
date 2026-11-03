# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_38_52
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

Report: generated/profiler/reports/2026_11_02_09_39_05/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.0       | 56.321   | 0.056  | 70.0       | 77.5          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9334.9     | 6914.765 | 6.915  | 9249.0     | 9328.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9382.0     | 6949.605 | 6.950  | 9182.0     | 9395.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9222.8     | 6831.704 | 6.832  | 5993.0     | 9340.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9230.8     | 6837.630 | 6.838  | 4376.0     | 9397.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9290.4     | 6881.802 | 6.882  | 8411.0     | 9307.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9397.1     | 6960.840 | 6.961  | 9266.0     | 9384.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9243.1     | 6846.741 | 6.847  | 6432.0     | 9343.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9247.8     | 6850.247 | 6.850  | 5078.0     | 9386.5        |

Final op time: 6.961 us (9397.1 cycles), device=5, bucket=NCRISC transport

