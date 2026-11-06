# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_40_23
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

Report: generated/profiler/reports/2026_11_04_05_40_33/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.1       | 56.395   | 0.056  | 70.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6810.2     | 5044.617 | 5.045  | 6704.0     | 6803.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6917.4     | 5124.000 | 5.124  | 6691.0     | 6855.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6752.3     | 5001.704 | 5.002  | 4696.0     | 6857.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6748.6     | 4998.988 | 4.999  | 3071.0     | 6872.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6795.2     | 5033.481 | 5.033  | 6420.0     | 6790.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6921.9     | 5127.333 | 5.127  | 6682.0     | 6844.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6760.9     | 5008.099 | 5.008  | 4722.0     | 6827.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6772.5     | 5016.667 | 5.017  | 3913.0     | 6860.0        |

Final op time: 5.127 us (6921.9 cycles), device=5, bucket=NCRISC transport

