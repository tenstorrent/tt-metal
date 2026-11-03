# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_24_33
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

### num_links=2, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_02_09_24_47/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.8       | 56.864   | 0.057  | 70.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9338.7     | 6917.580 | 6.918  | 9262.0     | 9335.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9366.6     | 6938.247 | 6.938  | 8410.0     | 9385.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9252.6     | 6853.753 | 6.854  | 6723.0     | 9334.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9249.9     | 6851.802 | 6.852  | 5040.0     | 9387.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9305.5     | 6892.938 | 6.893  | 8613.0     | 9313.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9388.0     | 6954.049 | 6.954  | 9199.0     | 9382.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9278.6     | 6873.062 | 6.873  | 7412.0     | 9314.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9285.0     | 6877.753 | 6.878  | 5919.0     | 9381.5        |

Final op time: 6.954 us (9388.0 cycles), device=5, bucket=NCRISC transport

