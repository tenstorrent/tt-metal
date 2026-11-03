# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_49_31
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

### num_links=1, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_02_09_49_43/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.9       | 55.506   | 0.056  | 74.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5064.1     | 3751.185 | 3.751  | 4972.0     | 5068.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5171.9     | 3831.037 | 3.831  | 5056.0     | 5139.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5076.8     | 3760.617 | 3.761  | 4356.0     | 5104.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5111.8     | 3786.543 | 3.787  | 4119.0     | 5135.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5092.7     | 3772.346 | 3.772  | 4945.0     | 5081.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5223.6     | 3869.309 | 3.869  | 4987.0     | 5153.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5113.6     | 3787.877 | 3.788  | 4932.0     | 5104.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5146.7     | 3812.370 | 3.812  | 4999.0     | 5151.5        |

Final op time: 3.869 us (5223.6 cycles), device=5, bucket=NCRISC transport

