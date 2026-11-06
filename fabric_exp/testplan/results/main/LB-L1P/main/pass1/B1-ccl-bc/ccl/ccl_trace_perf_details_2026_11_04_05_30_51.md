# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_30_51
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

Report: generated/profiler/reports/2026_11_04_05_31_02/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 79.1       | 58.568   | 0.059  | 70.0       | 81.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9546.8     | 7071.704 | 7.072  | 9379.0     | 9532.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9559.4     | 7081.012 | 7.081  | 8567.0     | 9568.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9454.1     | 7003.062 | 7.003  | 6733.0     | 9571.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9457.2     | 7005.309 | 7.005  | 5353.0     | 9586.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9499.1     | 7036.395 | 7.036  | 8843.0     | 9502.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9586.9     | 7101.383 | 7.101  | 9209.0     | 9570.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9462.6     | 7009.358 | 7.009  | 7128.0     | 9539.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9475.0     | 7018.519 | 7.019  | 6033.0     | 9614.5        |

Final op time: 7.101 us (9586.9 cycles), device=5, bucket=NCRISC transport

