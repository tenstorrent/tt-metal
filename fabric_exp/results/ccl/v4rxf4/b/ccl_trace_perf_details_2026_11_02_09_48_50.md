# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_48_50
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

### num_links=1, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_02_09_49_03/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.7       | 55.333   | 0.055  | 74.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6730.6     | 4985.630 | 4.986  | 6651.0     | 6733.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6833.1     | 5061.580 | 5.062  | 6738.0     | 6811.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6716.0     | 4974.815 | 4.975  | 5109.0     | 6763.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6721.2     | 4978.691 | 4.979  | 4026.0     | 6813.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6717.0     | 4975.580 | 4.976  | 6150.0     | 6730.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6847.9     | 5072.519 | 5.073  | 6706.0     | 6811.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6730.5     | 4985.556 | 4.986  | 5311.0     | 6769.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6736.3     | 4989.827 | 4.990  | 4499.0     | 6819.0        |

Final op time: 5.073 us (6847.9 cycles), device=5, bucket=NCRISC transport

