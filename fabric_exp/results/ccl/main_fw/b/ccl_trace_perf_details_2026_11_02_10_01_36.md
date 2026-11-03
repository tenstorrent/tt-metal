# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_01_36
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

Report: generated/profiler/reports/2026_11_02_10_01_47/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 73.8       | 54.642   | 0.055  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8459.2     | 6266.049 | 6.266  | 8298.0     | 8439.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8563.6     | 6343.407 | 6.343  | 8333.0     | 8524.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8426.1     | 6241.556 | 6.242  | 6742.0     | 8505.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8428.1     | 6243.012 | 6.243  | 5208.0     | 8559.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8448.0     | 6257.778 | 6.258  | 8131.0     | 8429.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8573.3     | 6350.593 | 6.351  | 8175.0     | 8519.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8430.2     | 6244.617 | 6.245  | 6810.0     | 8463.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8451.1     | 6260.099 | 6.260  | 6014.0     | 8533.5        |

Final op time: 6.351 us (8573.3 cycles), device=5, bucket=NCRISC transport

