# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_36_58
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

### num_links=1, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_02_09_37_11/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.6       | 55.235   | 0.055  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8388.0     | 6213.309 | 6.213  | 8327.0     | 8380.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8492.0     | 6290.346 | 6.290  | 8361.0     | 8477.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8367.8     | 6198.370 | 6.198  | 6819.0     | 8420.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8385.9     | 6211.778 | 6.212  | 5886.0     | 8465.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8387.5     | 6212.963 | 6.213  | 8123.0     | 8385.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8500.0     | 6296.272 | 6.296  | 8230.0     | 8464.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8368.0     | 6198.543 | 6.199  | 7072.0     | 8396.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8402.7     | 6224.198 | 6.224  | 6402.0     | 8472.0        |

Final op time: 6.296 us (8500.0 cycles), device=5, bucket=NCRISC transport

