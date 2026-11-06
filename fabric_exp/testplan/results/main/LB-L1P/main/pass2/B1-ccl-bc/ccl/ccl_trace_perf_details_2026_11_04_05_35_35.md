# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_35_35
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

### num_links=2, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_04_05_35_44/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 80.4       | 59.556   | 0.060  | 75.0       | 81.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12388.1    | 9176.370 | 9.176  | 12110.0    | 12407.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12392.7    | 9179.753 | 9.180  | 11660.0    | 12410.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12261.7    | 9082.765 | 9.083  | 9007.0     | 12374.5       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12257.3    | 9079.506 | 9.080  | 7288.0     | 12431.5       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12318.5    | 9124.815 | 9.125  | 11342.0    | 12364.5       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12403.1    | 9187.457 | 9.187  | 11873.0    | 12433.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12266.7    | 9086.469 | 9.086  | 9323.0     | 12391.5       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12254.4    | 9077.358 | 9.077  | 7524.0     | 12370.5       |

Final op time: 9.187 us (12403.1 cycles), device=5, bucket=NCRISC transport

