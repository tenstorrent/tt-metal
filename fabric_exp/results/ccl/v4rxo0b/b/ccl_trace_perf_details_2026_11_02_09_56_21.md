# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_56_21
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

### num_links=1, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_02_09_56_35/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.3       | 55.012   | 0.055  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5063.4     | 3750.691 | 3.751  | 5000.0     | 5068.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5192.8     | 3846.543 | 3.847  | 5079.0     | 5159.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5072.1     | 3757.086 | 3.757  | 4041.0     | 5097.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5100.7     | 3778.321 | 3.778  | 3602.0     | 5125.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5092.6     | 3772.296 | 3.772  | 4949.0     | 5089.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5227.6     | 3872.321 | 3.872  | 5016.0     | 5156.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5103.3     | 3780.247 | 3.780  | 4989.0     | 5099.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5141.0     | 3808.148 | 3.808  | 4920.0     | 5143.5        |

Final op time: 3.872 us (5227.6 cycles), device=5, bucket=NCRISC transport

