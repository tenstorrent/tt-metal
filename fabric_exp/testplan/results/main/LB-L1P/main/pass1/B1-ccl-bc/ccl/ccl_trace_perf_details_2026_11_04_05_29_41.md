# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_29_41
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

Report: generated/profiler/reports/2026_11_04_05_29_52/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.4       | 56.617   | 0.057  | 70.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6827.8     | 5057.654 | 5.058  | 6683.0     | 6809.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6937.4     | 5138.815 | 5.139  | 6669.0     | 6929.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6788.4     | 5028.469 | 5.028  | 4811.0     | 6876.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6789.1     | 5028.963 | 5.029  | 3410.0     | 6914.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6824.3     | 5055.062 | 5.055  | 6337.0     | 6800.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6941.2     | 5141.605 | 5.142  | 6549.0     | 6902.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6779.9     | 5022.173 | 5.022  | 4812.0     | 6852.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6802.9     | 5039.210 | 5.039  | 3810.0     | 6928.0        |

Final op time: 5.142 us (6941.2 cycles), device=5, bucket=NCRISC transport

