# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_35_45
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

### num_links=2, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_04_05_35_56/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 77.8       | 57.654   | 0.058  | 72.0       | 80.5          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10803.4    | 8002.494 | 8.002  | 10502.0    | 10836.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10821.1    | 8015.605 | 8.016  | 10228.0    | 10841.0       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10672.6    | 7905.605 | 7.906  | 7318.0     | 10793.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10642.3    | 7883.185 | 7.883  | 5033.0     | 10843.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10719.9    | 7940.691 | 7.941  | 9432.0     | 10800.0       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10839.6    | 8029.333 | 8.029  | 10387.0    | 10868.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10685.0    | 7914.815 | 7.915  | 7419.0     | 10795.0       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10665.2    | 7900.123 | 7.900  | 5373.0     | 10848.0       |

Final op time: 8.029 us (10839.6 cycles), device=5, bucket=NCRISC transport

