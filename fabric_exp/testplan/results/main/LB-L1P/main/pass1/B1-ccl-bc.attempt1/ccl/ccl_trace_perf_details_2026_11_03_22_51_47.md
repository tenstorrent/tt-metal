# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_51_47
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

Report: generated/profiler/reports/2026_11_03_22_51_58/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.4       | 56.617   | 0.057  | 73.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6907.7     | 5116.815 | 5.117  | 6693.0     | 6949.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 7001.0     | 5185.901 | 5.186  | 6781.0     | 7006.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6872.6     | 5090.790 | 5.091  | 4731.0     | 6945.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6875.8     | 5093.185 | 5.093  | 3413.0     | 6996.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6912.1     | 5120.049 | 5.120  | 6358.0     | 6941.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 7018.7     | 5199.062 | 5.199  | 6639.0     | 7000.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6885.4     | 5100.272 | 5.100  | 4912.0     | 6960.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6893.3     | 5106.173 | 5.106  | 3971.0     | 6985.0        |

Final op time: 5.199 us (7018.7 cycles), device=5, bucket=NCRISC transport

