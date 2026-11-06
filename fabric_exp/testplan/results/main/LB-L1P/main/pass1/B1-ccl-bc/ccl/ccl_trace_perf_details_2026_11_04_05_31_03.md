# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_31_03
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

Report: generated/profiler/reports/2026_11_04_05_31_13/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.8       | 56.889   | 0.057  | 71.0       | 76.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5487.0     | 4064.469 | 4.064  | 5333.0     | 5481.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5590.7     | 4141.235 | 4.141  | 5379.0     | 5530.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5414.5     | 4010.765 | 4.011  | 3919.0     | 5461.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5465.5     | 4048.494 | 4.048  | 3533.0     | 5518.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5472.1     | 4053.407 | 4.053  | 5118.0     | 5462.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5613.7     | 4158.296 | 4.158  | 5194.0     | 5536.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5455.9     | 4041.432 | 4.041  | 4992.0     | 5455.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5506.6     | 4078.963 | 4.079  | 4779.0     | 5526.5        |

Final op time: 4.158 us (5613.7 cycles), device=5, bucket=NCRISC transport

