# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_59_57
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

### num_links=1, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_03_23_00_11/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.7       | 56.790   | 0.057  | 75.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6802.6     | 5038.988 | 5.039  | 6663.0     | 6795.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6925.8     | 5130.198 | 5.130  | 6585.0     | 6899.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6765.1     | 5011.160 | 5.011  | 4857.0     | 6834.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6765.1     | 5011.185 | 5.011  | 3396.0     | 6886.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6808.3     | 5043.185 | 5.043  | 6561.0     | 6818.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6940.3     | 5140.938 | 5.141  | 6666.0     | 6875.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6775.0     | 5018.543 | 5.019  | 5204.0     | 6823.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6793.7     | 5032.395 | 5.032  | 4178.0     | 6905.5        |

Final op time: 5.141 us (6940.3 cycles), device=5, bucket=NCRISC transport

