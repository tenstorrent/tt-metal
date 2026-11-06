# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_40_00
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

Report: generated/profiler/reports/2026_11_04_05_40_11/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.0       | 56.296   | 0.056  | 70.0       | 75.5          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8523.0     | 6313.333 | 6.313  | 8351.0     | 8504.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8622.3     | 6386.864 | 6.387  | 8358.0     | 8613.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8455.9     | 6263.630 | 6.264  | 6077.0     | 8505.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8442.2     | 6253.481 | 6.253  | 4219.0     | 8563.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8501.4     | 6297.358 | 6.297  | 8158.0     | 8526.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8632.0     | 6394.099 | 6.394  | 8303.0     | 8577.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8450.6     | 6259.728 | 6.260  | 6242.0     | 8496.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8449.2     | 6258.642 | 6.259  | 4656.0     | 8558.0        |

Final op time: 6.394 us (8632.0 cycles), device=5, bucket=NCRISC transport

