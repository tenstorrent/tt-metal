# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_59_42
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

### num_links=1, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_03_22_59_55/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.6       | 56.741   | 0.057  | 73.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6867.0     | 5086.667 | 5.087  | 6689.0     | 6832.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 7000.0     | 5185.160 | 5.185  | 6697.0     | 6982.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6828.4     | 5058.074 | 5.058  | 4123.0     | 6923.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6802.6     | 5038.988 | 5.039  | 2145.0     | 6964.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6864.9     | 5085.111 | 5.085  | 6201.0     | 6850.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 7015.4     | 5196.617 | 5.197  | 6740.0     | 6974.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6821.1     | 5052.667 | 5.053  | 4303.0     | 6926.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6808.2     | 5043.111 | 5.043  | 2528.0     | 6963.5        |

Final op time: 5.197 us (7015.4 cycles), device=5, bucket=NCRISC transport

