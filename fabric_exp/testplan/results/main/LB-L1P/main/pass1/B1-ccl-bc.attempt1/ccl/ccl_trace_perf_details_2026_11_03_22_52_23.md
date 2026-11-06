# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_52_23
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

### num_links=1, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_03_22_52_34/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 78.5       | 58.148   | 0.058  | 71.0       | 79.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5119.4     | 3792.148 | 3.792  | 4961.0     | 5101.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5280.2     | 3911.284 | 3.911  | 4994.0     | 5213.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5108.3     | 3783.951 | 3.784  | 3344.0     | 5128.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5089.9     | 3770.272 | 3.770  | 1682.0     | 5181.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5133.8     | 3802.840 | 3.803  | 4909.0     | 5119.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5292.2     | 3920.173 | 3.920  | 4942.0     | 5181.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5118.0     | 3791.136 | 3.791  | 3548.0     | 5136.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5114.7     | 3788.691 | 3.789  | 2698.0     | 5160.5        |

Final op time: 3.920 us (5292.2 cycles), device=5, bucket=NCRISC transport

