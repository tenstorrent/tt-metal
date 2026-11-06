# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_29_29
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

Report: generated/profiler/reports/2026_11_04_05_29_40/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.4       | 56.593   | 0.057  | 73.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6851.4     | 5075.111 | 5.075  | 6688.0     | 6832.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6993.8     | 5180.593 | 5.181  | 6707.0     | 6923.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6834.3     | 5062.469 | 5.062  | 5004.0     | 6879.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6823.7     | 5054.568 | 5.055  | 3444.0     | 6942.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6859.1     | 5080.790 | 5.081  | 6400.0     | 6838.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6993.0     | 5180.000 | 5.180  | 6558.0     | 6959.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6832.0     | 5060.765 | 5.061  | 5110.0     | 6926.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6827.7     | 5057.531 | 5.058  | 3568.0     | 6959.0        |

Final op time: 5.181 us (6993.8 cycles), device=1, bucket=NCRISC transport

