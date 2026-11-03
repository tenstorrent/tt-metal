# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_17_09
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

Report: generated/profiler/reports/2026_11_02_09_17_19/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.0       | 56.321   | 0.056  | 70.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12389.5    | 9177.383 | 9.177  | 12169.0    | 12406.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12449.8    | 9222.049 | 9.222  | 12218.0    | 12469.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12298.7    | 9110.148 | 9.110  | 9112.0     | 12396.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12286.0    | 9100.741 | 9.101  | 7520.0     | 12455.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12354.0    | 9151.086 | 9.151  | 11615.0    | 12366.0       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12451.0    | 9222.988 | 9.223  | 12268.0    | 12453.5       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12321.8    | 9127.259 | 9.127  | 9612.0     | 12388.0       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12304.1    | 9114.123 | 9.114  | 8071.0     | 12445.0       |

Final op time: 9.223 us (12451.0 cycles), device=5, bucket=NCRISC transport

