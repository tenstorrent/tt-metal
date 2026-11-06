# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_41_19
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

Report: generated/profiler/reports/2026_11_04_05_41_29/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.7       | 56.815   | 0.057  | 71.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10728.4    | 7946.938 | 7.947  | 10591.0    | 10720.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10723.7    | 7943.481 | 7.943  | 9750.0     | 10750.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10619.8    | 7866.519 | 7.867  | 7925.0     | 10689.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10613.8    | 7862.074 | 7.862  | 6448.0     | 10727.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10661.1    | 7897.136 | 7.897  | 10048.0    | 10657.0       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10736.7    | 7953.136 | 7.953  | 10072.0    | 10730.5       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10633.1    | 7876.370 | 7.876  | 8303.0     | 10691.5       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10617.1    | 7864.519 | 7.865  | 6486.0     | 10779.0       |

Final op time: 7.953 us (10736.7 cycles), device=5, bucket=NCRISC transport

