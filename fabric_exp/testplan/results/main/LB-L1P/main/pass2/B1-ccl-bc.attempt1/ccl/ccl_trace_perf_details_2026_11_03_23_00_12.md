# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_23_00_12
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

Report: generated/profiler/reports/2026_11_03_23_00_26/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.9       | 55.457   | 0.055  | 72.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5591.5     | 4141.827 | 4.142  | 5461.0     | 5592.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5812.1     | 4305.235 | 4.305  | 5486.0     | 5645.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5474.9     | 4055.457 | 4.055  | 1920.0     | 5582.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5423.6     | 4017.481 | 4.017  | 175.0      | 5661.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5550.6     | 4111.531 | 4.112  | 4301.0     | 5584.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5769.3     | 4273.556 | 4.274  | 5440.0     | 5654.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5476.7     | 4056.815 | 4.057  | 1902.0     | 5617.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5451.0     | 4037.753 | 4.038  | 170.0      | 5664.0        |

Final op time: 4.305 us (5812.1 cycles), device=1, bucket=NCRISC transport

