# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_23_07
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

Report: generated/profiler/reports/2026_11_02_09_23_20/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 73.9       | 54.765   | 0.055  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6733.9     | 4988.074 | 4.988  | 6672.0     | 6728.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6858.3     | 5080.198 | 5.080  | 6729.0     | 6820.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6684.4     | 4951.432 | 4.951  | 4260.0     | 6786.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6659.2     | 4932.741 | 4.933  | 2292.0     | 6788.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6716.9     | 4975.506 | 4.976  | 6321.0     | 6725.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6857.7     | 5079.778 | 5.080  | 6652.0     | 6822.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6680.8     | 4948.765 | 4.949  | 4441.0     | 6755.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6669.3     | 4940.247 | 4.940  | 2746.0     | 6794.5        |

Final op time: 5.080 us (6858.3 cycles), device=1, bucket=NCRISC transport

