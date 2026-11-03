# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_39_06
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

Report: generated/profiler/reports/2026_11_02_09_39_20/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.2       | 56.469   | 0.056  | 73.0       | 76.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5409.3     | 4006.864 | 4.007  | 5358.0     | 5404.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5500.5     | 4074.444 | 4.074  | 5343.0     | 5474.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5374.8     | 3981.358 | 3.981  | 4906.0     | 5382.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5442.8     | 4031.728 | 4.032  | 4634.0     | 5463.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5416.8     | 4012.469 | 4.012  | 5267.0     | 5388.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5555.5     | 4115.185 | 4.115  | 5362.0     | 5472.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5413.9     | 4010.321 | 4.010  | 5214.0     | 5397.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5470.8     | 4052.420 | 4.052  | 5267.0     | 5471.0        |

Final op time: 4.115 us (5555.5 cycles), device=5, bucket=NCRISC transport

