# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_49_04
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

Report: generated/profiler/reports/2026_11_02_09_49_16/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 72.8       | 53.926   | 0.054  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6728.9     | 4984.395 | 4.984  | 6645.0     | 6733.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6818.3     | 5050.617 | 5.051  | 6681.0     | 6812.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6714.4     | 4973.654 | 4.974  | 5172.0     | 6758.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6710.6     | 4970.790 | 4.971  | 4015.0     | 6798.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6729.3     | 4984.691 | 4.985  | 6573.0     | 6752.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6835.1     | 5063.062 | 5.063  | 6655.0     | 6798.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6724.2     | 4980.914 | 4.981  | 5681.0     | 6756.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6725.7     | 4982.000 | 4.982  | 4587.0     | 6814.5        |

Final op time: 5.063 us (6835.1 cycles), device=5, bucket=NCRISC transport

