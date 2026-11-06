# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_35_57
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

Report: generated/profiler/reports/2026_11_04_05_36_07/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.4       | 56.593   | 0.057  | 70.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10765.8    | 7974.667 | 7.975  | 10597.0    | 10745.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10794.7    | 7996.049 | 7.996  | 10575.0    | 10774.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10620.6    | 7867.086 | 7.867  | 6936.0     | 10758.5       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10612.3    | 7860.988 | 7.861  | 5074.0     | 10812.5       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10681.7    | 7912.346 | 7.912  | 9278.0     | 10704.0       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10785.0    | 7988.864 | 7.989  | 10376.0    | 10825.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10636.7    | 7879.062 | 7.879  | 7265.0     | 10789.5       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10607.2    | 7857.160 | 7.857  | 4953.0     | 10819.0       |

Final op time: 7.996 us (10794.7 cycles), device=1, bucket=NCRISC transport

