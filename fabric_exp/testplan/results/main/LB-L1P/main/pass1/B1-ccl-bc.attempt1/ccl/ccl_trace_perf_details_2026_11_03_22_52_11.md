# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_52_11
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

Report: generated/profiler/reports/2026_11_03_22_52_22/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.5       | 55.160   | 0.055  | 72.0       | 72.5          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5617.3     | 4160.963 | 4.161  | 5497.0     | 5630.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5755.3     | 4263.185 | 4.263  | 5499.0     | 5659.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5550.1     | 4111.160 | 4.111  | 3253.0     | 5630.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5534.5     | 4099.654 | 4.100  | 1392.0     | 5691.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5605.4     | 4152.123 | 4.152  | 5284.0     | 5591.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5752.9     | 4261.383 | 4.261  | 5449.0     | 5695.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5559.6     | 4118.222 | 4.118  | 3503.0     | 5628.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5543.0     | 4105.926 | 4.106  | 1765.0     | 5647.0        |

Final op time: 4.263 us (5755.3 cycles), device=1, bucket=NCRISC transport

