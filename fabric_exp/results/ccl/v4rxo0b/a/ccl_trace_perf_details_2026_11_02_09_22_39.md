# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_22_39
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

### num_links=1, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_02_09_22_52/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.6       | 55.284   | 0.055  | 72.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8375.8     | 6204.296 | 6.204  | 8320.0     | 8370.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8463.0     | 6268.889 | 6.269  | 8375.0     | 8447.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8341.8     | 6179.086 | 6.179  | 6659.0     | 8405.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8354.8     | 6188.765 | 6.189  | 5375.0     | 8463.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8375.5     | 6204.049 | 6.204  | 8264.0     | 8375.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8491.6     | 6290.074 | 6.290  | 8304.0     | 8459.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8366.4     | 6197.333 | 6.197  | 7286.0     | 8407.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8388.0     | 6213.333 | 6.213  | 6317.0     | 8443.5        |

Final op time: 6.290 us (8491.6 cycles), device=5, bucket=NCRISC transport

