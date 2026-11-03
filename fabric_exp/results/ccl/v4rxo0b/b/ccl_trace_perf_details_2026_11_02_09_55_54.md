# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_55_54
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

### num_links=1, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_02_09_56_07/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 73.6       | 54.519   | 0.055  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6743.5     | 4995.185 | 4.995  | 6669.0     | 6738.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6845.1     | 5070.469 | 5.070  | 6729.0     | 6826.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6718.2     | 4976.444 | 4.976  | 4911.0     | 6776.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6724.7     | 4981.259 | 4.981  | 3928.0     | 6811.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6745.4     | 4996.617 | 4.997  | 6605.0     | 6745.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6857.5     | 5079.630 | 5.080  | 6702.0     | 6840.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6729.2     | 4984.617 | 4.985  | 5373.0     | 6761.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6747.4     | 4998.049 | 4.998  | 4773.0     | 6803.0        |

Final op time: 5.080 us (6857.5 cycles), device=5, bucket=NCRISC transport

