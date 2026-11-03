# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_23_21
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

Report: generated/profiler/reports/2026_11_02_09_23_35/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.3       | 55.012   | 0.055  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5523.5     | 4091.506 | 4.092  | 5447.0     | 5521.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5639.1     | 4177.086 | 4.177  | 5481.0     | 5602.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5486.5     | 4064.074 | 4.064  | 4052.0     | 5550.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5505.1     | 4077.852 | 4.078  | 2850.0     | 5586.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5526.1     | 4093.407 | 4.093  | 5216.0     | 5512.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5653.4     | 4187.704 | 4.188  | 5229.0     | 5599.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5509.1     | 4080.840 | 4.081  | 4800.0     | 5536.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5516.7     | 4086.420 | 4.086  | 3468.0     | 5589.0        |

Final op time: 4.188 us (5653.4 cycles), device=5, bucket=NCRISC transport

