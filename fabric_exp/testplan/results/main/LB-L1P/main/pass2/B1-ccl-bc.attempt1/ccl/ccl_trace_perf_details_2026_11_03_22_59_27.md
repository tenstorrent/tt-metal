# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_59_27
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

Report: generated/profiler/reports/2026_11_03_22_59_41/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.6       | 56.741   | 0.057  | 75.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8520.5     | 6311.457 | 6.311  | 8409.0     | 8500.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8670.2     | 6422.346 | 6.422  | 8344.0     | 8596.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8441.5     | 6252.938 | 6.253  | 5683.0     | 8528.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8409.3     | 6229.136 | 6.229  | 3328.0     | 8575.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8479.8     | 6281.358 | 6.281  | 7240.0     | 8500.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8640.6     | 6400.420 | 6.400  | 8206.0     | 8607.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8442.6     | 6253.753 | 6.254  | 5416.0     | 8546.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8434.5     | 6247.753 | 6.248  | 3478.0     | 8574.5        |

Final op time: 6.422 us (8670.2 cycles), device=1, bucket=NCRISC transport

