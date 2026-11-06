# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_41_30
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

Report: generated/profiler/reports/2026_11_04_05_41_40/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 78.6       | 58.247   | 0.058  | 71.0       | 81.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9521.2     | 7052.741 | 7.053  | 9282.0     | 9531.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9539.6     | 7066.395 | 7.066  | 8773.0     | 9571.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9428.7     | 6984.247 | 6.984  | 7086.0     | 9453.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9431.8     | 6986.543 | 6.987  | 5621.0     | 9554.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9468.6     | 7013.802 | 7.014  | 8982.0     | 9476.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9561.1     | 7082.296 | 7.082  | 9243.0     | 9569.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9470.9     | 7015.481 | 7.015  | 7892.0     | 9564.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9456.7     | 7004.938 | 7.005  | 6313.0     | 9589.0        |

Final op time: 7.082 us (9561.1 cycles), device=5, bucket=NCRISC transport

