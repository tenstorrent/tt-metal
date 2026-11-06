# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_29_53
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

### num_links=1, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_04_05_30_03/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.8       | 55.383   | 0.055  | 70.0       | 75.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5622.4     | 4164.716 | 4.165  | 5464.0     | 5617.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5736.0     | 4248.864 | 4.249  | 5509.0     | 5657.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5576.5     | 4130.716 | 4.131  | 3646.0     | 5663.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5565.1     | 4122.296 | 4.122  | 2038.0     | 5698.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5609.4     | 4155.111 | 4.155  | 5291.0     | 5594.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5742.1     | 4253.432 | 4.253  | 5453.0     | 5654.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5569.7     | 4125.728 | 4.126  | 3784.0     | 5630.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5583.3     | 4135.753 | 4.136  | 2695.0     | 5661.5        |

Final op time: 4.253 us (5742.1 cycles), device=5, bucket=NCRISC transport

