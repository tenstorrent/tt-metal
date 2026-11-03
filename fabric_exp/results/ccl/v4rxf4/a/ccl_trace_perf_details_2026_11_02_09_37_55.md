# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_37_55
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

Report: generated/profiler/reports/2026_11_02_09_38_08/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.8       | 55.432   | 0.055  | 74.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5050.6     | 3741.160 | 3.741  | 5003.0     | 5040.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5206.8     | 3856.889 | 3.857  | 5026.0     | 5126.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5033.7     | 3728.667 | 3.729  | 3378.0     | 5081.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5037.1     | 3731.210 | 3.731  | 2279.0     | 5122.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5069.4     | 3755.136 | 3.755  | 4798.0     | 5053.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5223.1     | 3868.988 | 3.869  | 5014.0     | 5122.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5058.5     | 3747.062 | 3.747  | 4260.0     | 5077.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5078.3     | 3761.704 | 3.762  | 3615.0     | 5117.0        |

Final op time: 3.869 us (5223.1 cycles), device=5, bucket=NCRISC transport

