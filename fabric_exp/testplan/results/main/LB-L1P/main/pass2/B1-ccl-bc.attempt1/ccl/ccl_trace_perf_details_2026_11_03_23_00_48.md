# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_23_00_48
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

### num_links=2, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_03_23_00_58/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 78.5       | 58.123   | 0.058  | 71.0       | 81.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10804.7    | 8003.481 | 8.003  | 10545.0    | 10841.0       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10794.3    | 7995.802 | 7.996  | 9479.0     | 10878.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10693.2    | 7920.889 | 7.921  | 7893.0     | 10802.5       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10689.8    | 7918.395 | 7.918  | 6160.0     | 10861.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10739.4    | 7955.136 | 7.955  | 9930.0     | 10725.5       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10818.8    | 8013.901 | 8.014  | 10203.0    | 10869.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10707.8    | 7931.704 | 7.932  | 8333.0     | 10805.0       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10700.0    | 7925.901 | 7.926  | 6650.0     | 10854.0       |

Final op time: 8.014 us (10818.8 cycles), device=5, bucket=NCRISC transport

