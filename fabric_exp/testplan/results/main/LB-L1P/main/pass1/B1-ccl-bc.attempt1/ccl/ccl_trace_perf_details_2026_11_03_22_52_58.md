# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_52_58
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

### num_links=2, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_03_22_53_09/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 75.5       | 55.926   | 0.056  | 71.0       | 75.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10802.1    | 8001.556 | 8.002  | 10623.0    | 10827.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10828.5    | 8021.136 | 8.021  | 10511.0    | 10844.0       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10667.3    | 7901.679 | 7.902  | 7270.0     | 10771.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10638.5    | 7880.346 | 7.880  | 5037.0     | 10829.5       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10720.5    | 7941.136 | 7.941  | 9477.0     | 10767.5       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10825.5    | 8018.889 | 8.019  | 10567.0    | 10840.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10670.1    | 7903.778 | 7.904  | 7481.0     | 10775.0       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10644.2    | 7884.617 | 7.885  | 5277.0     | 10829.5       |

Final op time: 8.021 us (10828.5 cycles), device=1, bucket=NCRISC transport

