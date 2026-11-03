# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_38_09
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

### num_links=2, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_02_09_38_23/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 75.9       | 56.198   | 0.056  | 70.0       | 77.5          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12265.8    | 9085.753 | 9.086  | 12165.0    | 12254.0       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12375.9    | 9167.358 | 9.167  | 12217.0    | 12315.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12132.6    | 8987.111 | 8.987  | 7847.0     | 12264.5       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12060.9    | 8934.000 | 8.934  | 4501.0     | 12334.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12193.0    | 9031.827 | 9.032  | 10551.0    | 12246.5       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12344.6    | 9144.173 | 9.144  | 12193.0    | 12321.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12123.0    | 8979.975 | 8.980  | 7511.0     | 12268.0       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12076.1    | 8945.235 | 8.945  | 5023.0     | 12319.5       |

Final op time: 9.167 us (12375.9 cycles), device=1, bucket=NCRISC transport

