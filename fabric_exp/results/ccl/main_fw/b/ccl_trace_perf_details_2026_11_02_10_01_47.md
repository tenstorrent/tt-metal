# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_01_47
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

Report: generated/profiler/reports/2026_11_02_10_01_57/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.7       | 55.333   | 0.055  | 72.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6790.5     | 5030.025 | 5.030  | 6704.0     | 6799.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6895.6     | 5107.827 | 5.108  | 6677.0     | 6870.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6758.4     | 5006.222 | 5.006  | 4636.0     | 6822.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6742.9     | 4994.741 | 4.995  | 2973.0     | 6856.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6777.7     | 5020.494 | 5.020  | 6432.0     | 6802.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6904.2     | 5114.222 | 5.114  | 6675.0     | 6878.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6762.3     | 5009.111 | 5.009  | 4773.0     | 6794.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6768.0     | 5013.358 | 5.013  | 3949.0     | 6842.0        |

Final op time: 5.114 us (6904.2 cycles), device=5, bucket=NCRISC transport

