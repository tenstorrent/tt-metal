# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_24_48
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

### num_links=2, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_02_09_24_59/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 75.1       | 55.605   | 0.056  | 70.0       | 75.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5417.9     | 4013.235 | 4.013  | 5376.0     | 5417.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5524.6     | 4092.272 | 4.092  | 5338.0     | 5478.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5354.8     | 3966.543 | 3.967  | 4010.0     | 5402.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5396.9     | 3997.679 | 3.998  | 3007.0     | 5469.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5394.4     | 3995.877 | 3.996  | 5281.0     | 5398.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5555.9     | 4115.506 | 4.116  | 5343.0     | 5470.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5378.2     | 3983.827 | 3.984  | 4737.0     | 5398.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5434.8     | 4025.802 | 4.026  | 4437.0     | 5454.0        |

Final op time: 4.116 us (5555.9 cycles), device=5, bucket=NCRISC transport

