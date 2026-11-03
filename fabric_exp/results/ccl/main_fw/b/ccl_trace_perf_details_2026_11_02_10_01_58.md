# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_01_58
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

### num_links=1, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_02_10_02_07/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.3       | 55.012   | 0.055  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6797.7     | 5035.333 | 5.035  | 6680.0     | 6806.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6894.3     | 5106.864 | 5.107  | 6624.0     | 6863.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6786.9     | 5027.358 | 5.027  | 5299.0     | 6814.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6798.8     | 5036.123 | 5.036  | 4432.0     | 6898.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6809.9     | 5044.370 | 5.044  | 6485.0     | 6815.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6916.4     | 5123.284 | 5.123  | 6696.0     | 6897.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6815.4     | 5048.420 | 5.048  | 5899.0     | 6854.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6827.0     | 5057.012 | 5.057  | 5195.0     | 6879.5        |

Final op time: 5.123 us (6916.4 cycles), device=5, bucket=NCRISC transport

