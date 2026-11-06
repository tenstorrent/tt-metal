# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_30_04
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

Report: generated/profiler/reports/2026_11_04_05_30_16/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 77.9       | 57.728   | 0.058  | 70.0       | 79.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5110.5     | 3785.531 | 3.786  | 4969.0     | 5083.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5245.7     | 3885.704 | 3.886  | 5033.0     | 5200.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5135.6     | 3804.148 | 3.804  | 3950.0     | 5152.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5138.7     | 3806.420 | 3.806  | 3256.0     | 5169.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5142.3     | 3809.111 | 3.809  | 4963.0     | 5108.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5277.4     | 3909.160 | 3.909  | 5060.0     | 5179.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5164.2     | 3825.358 | 3.825  | 4874.0     | 5131.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5166.1     | 3826.716 | 3.827  | 4040.0     | 5177.0        |

Final op time: 3.909 us (5277.4 cycles), device=5, bucket=NCRISC transport

