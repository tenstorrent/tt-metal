# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_35_01
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

Report: generated/profiler/reports/2026_11_04_05_35_12/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.7       | 56.840   | 0.057  | 75.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6842.7     | 5068.691 | 5.069  | 6663.0     | 6821.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6936.3     | 5137.975 | 5.138  | 6719.0     | 6938.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6812.6     | 5046.395 | 5.046  | 5120.0     | 6876.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6821.5     | 5052.963 | 5.053  | 3767.0     | 6919.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6848.2     | 5072.741 | 5.073  | 6476.0     | 6816.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6966.2     | 5160.173 | 5.160  | 6784.0     | 6937.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6826.2     | 5056.420 | 5.056  | 5339.0     | 6858.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6832.8     | 5061.309 | 5.061  | 4082.0     | 6915.5        |

Final op time: 5.160 us (6966.2 cycles), device=5, bucket=NCRISC transport

