# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_17_54
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

Report: generated/profiler/reports/2026_11_02_09_18_05/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 75.2       | 55.679   | 0.056  | 71.0       | 75.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5486.8     | 4064.272 | 4.064  | 5353.0     | 5480.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5602.9     | 4150.296 | 4.150  | 5330.0     | 5540.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5404.2     | 4003.111 | 4.003  | 3600.0     | 5445.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5432.0     | 4023.679 | 4.024  | 2202.0     | 5528.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5457.2     | 4042.370 | 4.042  | 5200.0     | 5443.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5613.8     | 4158.395 | 4.158  | 5278.0     | 5524.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5420.4     | 4015.086 | 4.015  | 4063.0     | 5444.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5452.3     | 4038.741 | 4.039  | 2902.0     | 5508.5        |

Final op time: 4.158 us (5613.8 cycles), device=5, bucket=NCRISC transport

