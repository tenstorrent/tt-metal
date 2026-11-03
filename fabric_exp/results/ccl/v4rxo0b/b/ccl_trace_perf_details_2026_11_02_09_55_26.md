# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_55_26
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

Report: generated/profiler/reports/2026_11_02_09_55_39/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.3       | 55.012   | 0.055  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8401.0     | 6222.963 | 6.223  | 8317.0     | 8400.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8506.5     | 6301.136 | 6.301  | 8380.0     | 8482.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8365.2     | 6196.469 | 6.196  | 6696.0     | 8407.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8389.5     | 6214.420 | 6.214  | 5778.0     | 8474.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8388.1     | 6213.407 | 6.213  | 7955.0     | 8392.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8516.8     | 6308.765 | 6.309  | 8373.0     | 8498.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8377.0     | 6205.185 | 6.205  | 6926.0     | 8411.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8396.2     | 6219.407 | 6.219  | 6033.0     | 8465.0        |

Final op time: 6.309 us (8516.8 cycles), device=5, bucket=NCRISC transport

