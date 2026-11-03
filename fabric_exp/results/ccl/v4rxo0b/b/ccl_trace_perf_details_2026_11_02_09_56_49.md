# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_56_49
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

### num_links=2, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_02_09_57_03/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.7       | 56.840   | 0.057  | 70.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10558.7    | 7821.284 | 7.821  | 10461.0    | 10545.0       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10567.6    | 7827.877 | 7.828  | 9273.0     | 10627.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10476.2    | 7760.173 | 7.760  | 8039.0     | 10543.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10478.8    | 7762.074 | 7.762  | 6610.0     | 10607.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10527.4    | 7798.049 | 7.798  | 9844.0     | 10552.5       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10596.2    | 7849.062 | 7.849  | 9991.0     | 10630.5       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10500.0    | 7777.802 | 7.778  | 8499.0     | 10551.5       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10512.5    | 7787.062 | 7.787  | 7412.0     | 10601.0       |

Final op time: 7.849 us (10596.2 cycles), device=5, bucket=NCRISC transport

