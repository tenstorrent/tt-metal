# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_02_52
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

### num_links=2, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_02_10_03_01/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.6       | 56.716   | 0.057  | 70.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10774.4    | 7981.012 | 7.981  | 10544.0    | 10806.0       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10794.3    | 7995.753 | 7.996  | 9652.0     | 10857.0       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10680.4    | 7911.407 | 7.911  | 7656.0     | 10780.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10677.7    | 7909.407 | 7.909  | 6188.0     | 10820.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10738.8    | 7954.667 | 7.955  | 9961.0     | 10788.5       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10815.7    | 8011.654 | 8.012  | 10265.0    | 10835.5       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10690.0    | 7918.543 | 7.919  | 8069.0     | 10770.0       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10708.5    | 7932.198 | 7.932  | 6985.0     | 10825.0       |

Final op time: 8.012 us (10815.7 cycles), device=5, bucket=NCRISC transport

