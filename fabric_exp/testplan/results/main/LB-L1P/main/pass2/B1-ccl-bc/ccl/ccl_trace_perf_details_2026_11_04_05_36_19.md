# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_36_19
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

Report: generated/profiler/reports/2026_11_04_05_36_30/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.1       | 56.395   | 0.056  | 71.0       | 76.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5505.0     | 4077.778 | 4.078  | 5386.0     | 5485.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5571.7     | 4127.160 | 4.127  | 5400.0     | 5536.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5441.0     | 4030.346 | 4.030  | 4639.0     | 5458.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5495.9     | 4071.062 | 4.071  | 4194.0     | 5524.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5489.8     | 4066.494 | 4.066  | 5260.0     | 5458.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5618.9     | 4162.123 | 4.162  | 5337.0     | 5525.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5481.2     | 4060.123 | 4.060  | 4929.0     | 5465.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5533.9     | 4099.185 | 4.099  | 5184.0     | 5530.5        |

Final op time: 4.162 us (5618.9 cycles), device=5, bucket=NCRISC transport

