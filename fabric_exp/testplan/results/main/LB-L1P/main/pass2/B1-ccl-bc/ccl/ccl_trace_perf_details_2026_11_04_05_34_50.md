# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_34_50
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

Report: generated/profiler/reports/2026_11_04_05_35_01/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.9       | 56.938   | 0.057  | 75.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6873.8     | 5091.704 | 5.092  | 6701.0     | 6853.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6963.1     | 5157.877 | 5.158  | 6762.0     | 6958.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6862.2     | 5083.136 | 5.083  | 5296.0     | 6921.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6865.6     | 5085.654 | 5.086  | 4122.0     | 6959.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6893.9     | 5106.617 | 5.107  | 6430.0     | 6940.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6969.2     | 5162.370 | 5.162  | 6695.0     | 6973.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6869.9     | 5088.840 | 5.089  | 5775.0     | 6882.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6881.9     | 5097.679 | 5.098  | 4759.0     | 6961.5        |

Final op time: 5.162 us (6969.2 cycles), device=5, bucket=NCRISC transport

