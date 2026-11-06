# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_40_56
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

Report: generated/profiler/reports/2026_11_04_05_41_07/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 79.1       | 58.593   | 0.059  | 70.0       | 81.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12371.5    | 9164.049 | 9.164  | 12199.0    | 12374.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12391.8    | 9179.111 | 9.179  | 11541.0    | 12388.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12257.8    | 9079.827 | 9.080  | 8923.0     | 12387.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12243.3    | 9069.136 | 9.069  | 7140.0     | 12421.5       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12298.5    | 9110.025 | 9.110  | 11166.0    | 12347.5       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12394.3    | 9180.938 | 9.181  | 11699.0    | 12394.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12270.1    | 9088.963 | 9.089  | 9073.0     | 12391.5       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12247.1    | 9071.951 | 9.072  | 7194.0     | 12430.0       |

Final op time: 9.181 us (12394.3 cycles), device=5, bucket=NCRISC transport

