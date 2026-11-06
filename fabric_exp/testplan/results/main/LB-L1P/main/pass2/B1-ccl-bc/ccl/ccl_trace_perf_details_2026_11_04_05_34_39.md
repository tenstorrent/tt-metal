# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_34_39
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

### num_links=1, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_04_05_34_49/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.2       | 56.444   | 0.056  | 75.0       | 76.5          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8533.8     | 6321.333 | 6.321  | 8399.0     | 8538.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8622.4     | 6386.938 | 6.387  | 8361.0     | 8608.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8509.0     | 6302.988 | 6.303  | 7360.0     | 8540.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8530.8     | 6319.111 | 6.319  | 6365.0     | 8604.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8536.6     | 6323.432 | 6.323  | 8300.0     | 8526.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8644.6     | 6403.407 | 6.403  | 8422.0     | 8600.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8524.3     | 6314.272 | 6.314  | 7735.0     | 8554.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8542.5     | 6327.802 | 6.328  | 6737.0     | 8621.5        |

Final op time: 6.403 us (8644.6 cycles), device=5, bucket=NCRISC transport

