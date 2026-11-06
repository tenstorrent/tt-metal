# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_53_10
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

### num_links=2, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_03_22_53_21/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 78.3       | 58.000   | 0.058  | 71.0       | 81.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9537.4     | 7064.765 | 7.065  | 9287.0     | 9533.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9570.8     | 7089.481 | 7.089  | 9215.0     | 9589.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9420.8     | 6978.395 | 6.978  | 6120.0     | 9558.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9413.0     | 6972.593 | 6.973  | 4639.0     | 9556.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9486.8     | 7027.284 | 7.027  | 8615.0     | 9486.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9582.1     | 7097.827 | 7.098  | 9365.0     | 9575.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9437.3     | 6990.568 | 6.991  | 6710.0     | 9522.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9429.9     | 6985.111 | 6.985  | 5060.0     | 9564.5        |

Final op time: 7.098 us (9582.1 cycles), device=5, bucket=NCRISC transport

