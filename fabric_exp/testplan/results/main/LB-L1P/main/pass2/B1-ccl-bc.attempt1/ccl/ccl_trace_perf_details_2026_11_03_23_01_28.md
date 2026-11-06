# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_23_01_28
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

### num_links=2, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_03_23_01_41/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.6       | 56.741   | 0.057  | 71.0       | 76.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5490.0     | 4066.691 | 4.067  | 5357.0     | 5490.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5619.4     | 4162.543 | 4.163  | 5401.0     | 5529.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5400.5     | 4000.370 | 4.000  | 3755.0     | 5435.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5441.1     | 4030.469 | 4.030  | 2879.0     | 5529.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5478.4     | 4058.074 | 4.058  | 5307.0     | 5471.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5612.2     | 4157.210 | 4.157  | 5328.0     | 5516.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5418.6     | 4013.753 | 4.014  | 4441.0     | 5430.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5486.8     | 4064.296 | 4.064  | 4345.0     | 5531.0        |

Final op time: 4.163 us (5619.4 cycles), device=1, bucket=NCRISC transport

