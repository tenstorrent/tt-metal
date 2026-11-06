# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_40_34
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

Report: generated/profiler/reports/2026_11_04_05_40_44/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.8       | 55.432   | 0.055  | 72.0       | 76.5          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5605.4     | 4152.148 | 4.152  | 5458.0     | 5601.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5698.8     | 4221.333 | 4.221  | 5513.0     | 5665.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5573.3     | 4128.395 | 4.128  | 4181.0     | 5588.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5598.6     | 4147.086 | 4.147  | 3361.0     | 5694.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5605.0     | 4151.827 | 4.152  | 5412.0     | 5566.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5721.2     | 4237.926 | 4.238  | 5475.0     | 5653.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5601.4     | 4149.185 | 4.149  | 4861.0     | 5608.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5627.3     | 4168.346 | 4.168  | 4542.0     | 5647.0        |

Final op time: 4.238 us (5721.2 cycles), device=5, bucket=NCRISC transport

