# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_50_39
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

Report: generated/profiler/reports/2026_11_02_09_50_48/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 75.2       | 55.728   | 0.056  | 70.0       | 75.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5425.1     | 4018.617 | 4.019  | 5336.0     | 5424.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5533.1     | 4098.593 | 4.099  | 5397.0     | 5475.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5365.5     | 3974.444 | 3.974  | 4133.0     | 5406.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5400.7     | 4000.519 | 4.001  | 2949.0     | 5494.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5415.9     | 4011.778 | 4.012  | 5294.0     | 5410.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5565.5     | 4122.617 | 4.123  | 5287.0     | 5493.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5395.0     | 3996.296 | 3.996  | 4933.0     | 5405.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5446.9     | 4034.716 | 4.035  | 4358.0     | 5472.0        |

Final op time: 4.123 us (5565.5 cycles), device=5, bucket=NCRISC transport

