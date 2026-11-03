# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_37_12
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

Report: generated/profiler/reports/2026_11_02_09_37_25/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.5       | 55.160   | 0.055  | 72.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6755.2     | 5003.827 | 5.004  | 6645.0     | 6754.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6860.6     | 5081.951 | 5.082  | 6732.0     | 6835.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6743.3     | 4995.012 | 4.995  | 5196.0     | 6782.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6748.8     | 4999.086 | 4.999  | 4177.0     | 6840.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6752.7     | 5002.025 | 5.002  | 6486.0     | 6749.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6863.3     | 5083.901 | 5.084  | 6712.0     | 6835.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6751.7     | 5001.235 | 5.001  | 5356.0     | 6801.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6764.5     | 5010.765 | 5.011  | 4673.0     | 6843.0        |

Final op time: 5.084 us (6863.3 cycles), device=5, bucket=NCRISC transport

