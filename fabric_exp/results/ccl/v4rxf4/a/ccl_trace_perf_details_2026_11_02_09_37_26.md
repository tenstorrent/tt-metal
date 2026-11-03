# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_37_26
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

Report: generated/profiler/reports/2026_11_02_09_37_39/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 73.6       | 54.494   | 0.054  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6731.7     | 4986.420 | 4.986  | 6653.0     | 6734.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6839.0     | 5065.926 | 5.066  | 6610.0     | 6814.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6695.6     | 4959.679 | 4.960  | 4450.0     | 6777.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6691.2     | 4956.469 | 4.956  | 3123.0     | 6805.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6724.3     | 4980.938 | 4.981  | 6486.0     | 6734.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6845.6     | 5070.840 | 5.071  | 6568.0     | 6816.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6706.1     | 4967.506 | 4.968  | 4816.0     | 6780.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6695.1     | 4959.358 | 4.959  | 3522.0     | 6808.0        |

Final op time: 5.071 us (6845.6 cycles), device=5, bucket=NCRISC transport

