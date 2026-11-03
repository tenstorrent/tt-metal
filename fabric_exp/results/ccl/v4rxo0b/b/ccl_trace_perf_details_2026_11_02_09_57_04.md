# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_57_04
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

Report: generated/profiler/reports/2026_11_02_09_57_16/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.5       | 56.642   | 0.057  | 70.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10600.5    | 7852.247 | 7.852  | 10529.0    | 10596.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10613.3    | 7861.704 | 7.862  | 9183.0     | 10662.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10513.5    | 7787.753 | 7.788  | 7709.0     | 10596.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10509.6    | 7784.889 | 7.785  | 6030.0     | 10647.5       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10564.1    | 7825.259 | 7.825  | 9693.0     | 10599.5       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10635.1    | 7877.877 | 7.878  | 9841.0     | 10646.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10522.3    | 7794.272 | 7.794  | 7907.0     | 10609.0       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10510.5    | 7785.556 | 7.786  | 6135.0     | 10676.5       |

Final op time: 7.878 us (10635.1 cycles), device=5, bucket=NCRISC transport

