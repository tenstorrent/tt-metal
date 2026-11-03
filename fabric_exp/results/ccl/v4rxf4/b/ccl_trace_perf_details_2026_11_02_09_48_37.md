# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_48_37
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

Report: generated/profiler/reports/2026_11_02_09_48_49/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.8       | 55.432   | 0.055  | 72.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8394.9     | 6218.469 | 6.218  | 8299.0     | 8399.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8495.7     | 6293.111 | 6.293  | 8367.0     | 8477.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8363.0     | 6194.840 | 6.195  | 6460.0     | 8421.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8381.6     | 6208.617 | 6.209  | 5523.0     | 8470.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8390.8     | 6215.432 | 6.215  | 8090.0     | 8390.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8509.1     | 6303.037 | 6.303  | 8289.0     | 8490.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8372.2     | 6201.630 | 6.202  | 6986.0     | 8411.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 8394.7     | 6218.296 | 6.218  | 6054.0     | 8462.0        |

Final op time: 6.303 us (8509.1 cycles), device=5, bucket=NCRISC transport

