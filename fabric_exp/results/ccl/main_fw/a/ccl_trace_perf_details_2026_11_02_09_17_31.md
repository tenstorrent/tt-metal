# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_17_31
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

Report: generated/profiler/reports/2026_11_02_09_17_41/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 77.2       | 57.185   | 0.057  | 70.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10746.6    | 7960.469 | 7.960  | 10577.0    | 10720.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10780.6    | 7985.605 | 7.986  | 9844.0     | 10823.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10656.7    | 7893.852 | 7.894  | 7641.0     | 10773.5       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10643.9    | 7884.370 | 7.884  | 5731.0     | 10808.5       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10718.2    | 7939.383 | 7.939  | 10014.0    | 10786.0       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10795.4    | 7996.617 | 7.997  | 10326.0    | 10819.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10667.3    | 7901.679 | 7.902  | 7970.0     | 10770.5       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10670.7    | 7904.198 | 7.904  | 6534.0     | 10818.0       |

Final op time: 7.997 us (10795.4 cycles), device=5, bucket=NCRISC transport

