# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_16_12
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

Report: generated/profiler/reports/2026_11_02_09_16_23/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.3       | 55.012   | 0.055  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8452.3     | 6260.988 | 6.261  | 8281.0     | 8442.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8557.2     | 6338.667 | 6.339  | 8305.0     | 8526.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8418.4     | 6235.852 | 6.236  | 6716.0     | 8409.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8423.7     | 6239.802 | 6.240  | 5557.0     | 8468.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8452.8     | 6261.358 | 6.261  | 8051.0     | 8483.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8570.4     | 6348.469 | 6.348  | 8191.0     | 8522.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8433.0     | 6246.667 | 6.247  | 7096.0     | 8495.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8444.8     | 6255.407 | 6.255  | 6012.0     | 8527.5        |

Final op time: 6.348 us (8570.4 cycles), device=5, bucket=NCRISC transport

