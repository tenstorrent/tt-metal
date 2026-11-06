# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_36_08
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

Report: generated/profiler/reports/2026_11_04_05_36_19/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 79.2       | 58.667   | 0.059  | 72.0       | 81.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9546.6     | 7071.556 | 7.072  | 9301.0     | 9534.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9564.1     | 7084.543 | 7.085  | 8768.0     | 9583.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9453.4     | 7002.519 | 7.003  | 6475.0     | 9583.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9432.6     | 6987.086 | 6.987  | 4699.0     | 9557.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9493.6     | 7032.272 | 7.032  | 8799.0     | 9515.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9592.2     | 7105.333 | 7.105  | 9213.0     | 9577.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9465.0     | 7011.111 | 7.011  | 7198.0     | 9556.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9465.6     | 7011.556 | 7.012  | 5842.0     | 9625.5        |

Final op time: 7.105 us (9592.2 cycles), device=5, bucket=NCRISC transport

