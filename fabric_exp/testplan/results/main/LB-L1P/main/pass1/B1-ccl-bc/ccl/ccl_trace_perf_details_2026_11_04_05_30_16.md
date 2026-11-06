# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_30_16
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

Report: generated/profiler/reports/2026_11_04_05_30_27/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 79.1       | 58.617   | 0.059  | 71.0       | 81.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12386.7    | 9175.309 | 9.175  | 12149.0    | 12405.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12427.2    | 9205.309 | 9.205  | 12156.0    | 12448.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12256.2    | 9078.691 | 9.079  | 8885.0     | 12363.5       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12252.3    | 9075.778 | 9.076  | 7092.0     | 12414.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12326.4    | 9130.691 | 9.131  | 11289.0    | 12353.0       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12405.2    | 9189.037 | 9.189  | 12039.0    | 12459.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12278.7    | 9095.333 | 9.095  | 9396.0     | 12404.5       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12238.4    | 9065.481 | 9.065  | 6962.0     | 12363.5       |

Final op time: 9.205 us (12427.2 cycles), device=1, bucket=NCRISC transport

