# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_23_01_12
CCLs: broadcast

## broadcast

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_broadcast.py::test_ccl_broadcast
Configured benchmark shape: [1, 7168]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = slowest device's slowest top-level bucket, using per-device averages across iterations only.

Cross-device bucket averages are omitted for this CCL.

### num_links=2, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_03_23_01_27/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 79.6       | 58.938   | 0.059  | 72.0       | 81.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9494.2     | 7032.741 | 7.033  | 9268.0     | 9512.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9548.1     | 7072.667 | 7.073  | 9196.0     | 9574.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9376.8     | 6945.753 | 6.946  | 6009.0     | 9504.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9330.2     | 6911.235 | 6.911  | 3316.0     | 9597.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9435.1     | 6988.963 | 6.989  | 8418.0     | 9456.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9560.1     | 7081.556 | 7.082  | 9207.0     | 9558.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9387.3     | 6953.580 | 6.954  | 6029.0     | 9485.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9361.0     | 6934.074 | 6.934  | 3981.0     | 9506.0        |

Final op time: 7.082 us (9560.1 cycles), device=5, bucket=NCRISC transport

