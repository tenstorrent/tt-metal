# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_02_30
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

Report: generated/profiler/reports/2026_11_02_10_02_40/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 75.2       | 55.704   | 0.056  | 70.0       | 75.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12383.0    | 9172.568 | 9.173  | 12247.0    | 12402.0       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12439.1    | 9214.148 | 9.214  | 12288.0    | 12452.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12260.4    | 9081.778 | 9.082  | 8551.0     | 12393.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12212.1    | 9045.975 | 9.046  | 5938.0     | 12415.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12319.2    | 9125.333 | 9.125  | 11000.0    | 12357.0       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12427.1    | 9205.235 | 9.205  | 12146.0    | 12428.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12249.1    | 9073.383 | 9.073  | 8406.0     | 12381.5       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12222.3    | 9053.556 | 9.054  | 6224.0     | 12415.0       |

Final op time: 9.214 us (12439.1 cycles), device=1, bucket=NCRISC transport

