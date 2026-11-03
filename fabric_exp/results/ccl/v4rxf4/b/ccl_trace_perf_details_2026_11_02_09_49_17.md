# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_49_17
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

### num_links=1, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_02_09_49_30/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 73.2       | 54.222   | 0.054  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5524.1     | 4091.951 | 4.092  | 5447.0     | 5516.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5622.1     | 4164.494 | 4.164  | 5506.0     | 5600.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5512.5     | 4083.358 | 4.083  | 4438.0     | 5546.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5530.6     | 4096.765 | 4.097  | 3493.0     | 5596.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5536.0     | 4100.741 | 4.101  | 5385.0     | 5536.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5658.4     | 4191.407 | 4.191  | 5456.0     | 5606.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5523.8     | 4091.679 | 4.092  | 4950.0     | 5537.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5566.0     | 4122.938 | 4.123  | 4736.0     | 5592.5        |

Final op time: 4.191 us (5658.4 cycles), device=5, bucket=NCRISC transport

