# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_56_36
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

Report: generated/profiler/reports/2026_11_02_09_56_49/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.1       | 56.370   | 0.056  | 70.0       | 75.5          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12261.0    | 9082.198 | 9.082  | 12164.0    | 12242.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12273.2    | 9091.235 | 9.091  | 10937.0    | 12309.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12183.5    | 9024.815 | 9.025  | 9479.0     | 12266.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12170.2    | 9014.938 | 9.015  | 7809.0     | 12319.5       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12204.3    | 9040.222 | 9.040  | 11090.0    | 12235.5       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12287.8    | 9102.074 | 9.102  | 11532.0    | 12299.5       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12190.2    | 9029.753 | 9.030  | 9753.0     | 12251.0       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12182.2    | 9023.827 | 9.024  | 8326.0     | 12298.0       |

Final op time: 9.102 us (12287.8 cycles), device=5, bucket=NCRISC transport

