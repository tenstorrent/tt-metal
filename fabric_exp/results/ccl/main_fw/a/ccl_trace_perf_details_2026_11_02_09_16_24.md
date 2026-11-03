# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_16_24
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

### num_links=1, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_02_09_16_35/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.6       | 55.284   | 0.055  | 72.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6810.8     | 5045.037 | 5.045  | 6642.0     | 6830.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6956.2     | 5152.716 | 5.153  | 6595.0     | 6889.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6764.8     | 5010.988 | 5.011  | 4192.0     | 6836.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6727.3     | 4983.210 | 4.983  | 1917.0     | 6866.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6797.5     | 5035.185 | 5.035  | 6265.0     | 6815.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6957.1     | 5153.407 | 5.153  | 6629.0     | 6896.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6755.1     | 5003.753 | 5.004  | 4101.0     | 6813.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6752.4     | 5001.778 | 5.002  | 2695.0     | 6879.0        |

Final op time: 5.153 us (6957.1 cycles), device=5, bucket=NCRISC transport

