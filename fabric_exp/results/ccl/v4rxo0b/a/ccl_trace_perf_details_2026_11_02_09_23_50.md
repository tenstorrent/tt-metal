# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_23_50
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

### num_links=2, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_02_09_24_03/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 75.5       | 55.901   | 0.056  | 70.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12259.9    | 9081.383 | 9.081  | 12158.0    | 12251.0       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12295.8    | 9108.000 | 9.108  | 11728.0    | 12309.0       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12150.6    | 9000.420 | 9.000  | 8591.0     | 12275.5       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12120.1    | 8977.877 | 8.978  | 6581.0     | 12310.5       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12202.0    | 9038.519 | 9.039  | 11080.0    | 12224.5       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12301.7    | 9112.395 | 9.112  | 12079.0    | 12303.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12162.1    | 9008.963 | 9.009  | 9086.0     | 12243.5       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12124.6    | 8981.185 | 8.981  | 6805.0     | 12299.0       |

Final op time: 9.112 us (12301.7 cycles), device=5, bucket=NCRISC transport

