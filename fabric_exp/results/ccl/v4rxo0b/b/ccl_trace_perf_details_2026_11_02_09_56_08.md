# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_56_08
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

Report: generated/profiler/reports/2026_11_02_09_56_21/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.5       | 55.210   | 0.055  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5512.3     | 4083.185 | 4.083  | 5450.0     | 5504.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5617.3     | 4160.988 | 4.161  | 5486.0     | 5602.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5484.5     | 4062.568 | 4.063  | 4086.0     | 5515.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5509.4     | 4081.037 | 4.081  | 3116.0     | 5584.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5517.3     | 4086.864 | 4.087  | 5342.0     | 5523.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5649.9     | 4185.111 | 4.185  | 5421.0     | 5603.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5512.3     | 4083.210 | 4.083  | 4721.0     | 5537.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5539.5     | 4103.309 | 4.103  | 4154.0     | 5574.5        |

Final op time: 4.185 us (5649.9 cycles), device=5, bucket=NCRISC transport

