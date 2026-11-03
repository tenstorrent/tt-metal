# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_03_13
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

### num_links=2, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_02_10_03_23/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.1       | 54.914   | 0.055  | 70.0       | 75.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5469.9     | 4051.802 | 4.052  | 5352.0     | 5473.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5563.4     | 4121.037 | 4.121  | 5396.0     | 5526.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5424.6     | 4018.198 | 4.018  | 4647.0     | 5437.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5480.3     | 4059.457 | 4.059  | 4308.0     | 5518.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5465.2     | 4048.321 | 4.048  | 5215.0     | 5449.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5605.0     | 4151.877 | 4.152  | 5285.0     | 5528.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5460.0     | 4044.469 | 4.044  | 5204.0     | 5439.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5522.8     | 4090.988 | 4.091  | 5297.0     | 5524.0        |

Final op time: 4.152 us (5605.0 cycles), device=5, bucket=NCRISC transport

