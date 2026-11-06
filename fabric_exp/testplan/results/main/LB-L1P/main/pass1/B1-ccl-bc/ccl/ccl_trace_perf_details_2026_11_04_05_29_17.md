# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_29_17
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

Report: generated/profiler/reports/2026_11_04_05_29_28/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.2       | 56.420   | 0.056  | 73.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8526.3     | 6315.778 | 6.316  | 8395.0     | 8507.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8620.4     | 6385.457 | 6.385  | 8362.0     | 8599.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8450.3     | 6259.481 | 6.259  | 6247.0     | 8500.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8450.1     | 6259.333 | 6.259  | 4578.0     | 8578.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8500.7     | 6296.840 | 6.297  | 8155.0     | 8526.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8625.8     | 6389.481 | 6.389  | 8330.0     | 8582.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8466.5     | 6271.457 | 6.271  | 6416.0     | 8529.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8462.6     | 6268.593 | 6.269  | 4866.0     | 8580.5        |

Final op time: 6.389 us (8625.8 cycles), device=5, bucket=NCRISC transport

