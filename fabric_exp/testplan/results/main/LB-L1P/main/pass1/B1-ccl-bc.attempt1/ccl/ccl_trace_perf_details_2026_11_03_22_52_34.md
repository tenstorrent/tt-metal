# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_52_34
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

Report: generated/profiler/reports/2026_11_03_22_52_45/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 79.2       | 58.667   | 0.059  | 72.0       | 81.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12439.8    | 9214.691 | 9.215  | 12275.0    | 12447.0       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12457.0    | 9227.383 | 9.227  | 11764.0    | 12486.0       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12327.7    | 9131.630 | 9.132  | 9459.0     | 12421.5       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12319.1    | 9125.284 | 9.125  | 7738.0     | 12450.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12386.5    | 9175.185 | 9.175  | 11571.0    | 12384.0       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12458.5    | 9228.543 | 9.229  | 11858.0    | 12480.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12355.2    | 9152.025 | 9.152  | 9864.0     | 12422.5       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12341.2    | 9141.630 | 9.142  | 8237.0     | 12491.5       |

Final op time: 9.229 us (12458.5 cycles), device=5, bucket=NCRISC transport

