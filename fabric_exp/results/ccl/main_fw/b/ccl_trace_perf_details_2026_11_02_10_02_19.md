# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_02_19
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

Report: generated/profiler/reports/2026_11_02_10_02_29/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.6       | 55.284   | 0.055  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5101.8     | 3779.111 | 3.779  | 4964.0     | 5090.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5230.1     | 3874.173 | 3.874  | 5067.0     | 5185.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5105.8     | 3782.099 | 3.782  | 4034.0     | 5134.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5127.6     | 3798.247 | 3.798  | 3369.0     | 5157.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5111.1     | 3786.025 | 3.786  | 4959.0     | 5093.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5261.7     | 3897.531 | 3.898  | 4870.0     | 5185.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5138.4     | 3806.198 | 3.806  | 4879.0     | 5131.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5166.3     | 3826.914 | 3.827  | 4620.0     | 5171.5        |

Final op time: 3.898 us (5261.7 cycles), device=5, bucket=NCRISC transport

