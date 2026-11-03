# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_02_08
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

Report: generated/profiler/reports/2026_11_02_10_02_18/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.2       | 54.988   | 0.055  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5569.7     | 4125.728 | 4.126  | 5456.0     | 5592.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5696.0     | 4219.259 | 4.219  | 5447.0     | 5651.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5541.7     | 4104.988 | 4.105  | 3941.0     | 5581.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5562.2     | 4120.173 | 4.120  | 2958.0     | 5648.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5580.5     | 4133.679 | 4.134  | 5417.0     | 5552.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5702.4     | 4224.000 | 4.224  | 5344.0     | 5648.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5551.0     | 4111.827 | 4.112  | 4412.0     | 5564.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5598.1     | 4146.741 | 4.147  | 4050.0     | 5623.5        |

Final op time: 4.224 us (5702.4 cycles), device=5, bucket=NCRISC transport

