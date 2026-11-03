# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_16_58
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

### num_links=1, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_02_09_17_08/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.5       | 55.210   | 0.055  | 72.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5088.5     | 3769.259 | 3.769  | 4971.0     | 5086.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5201.8     | 3853.160 | 3.853  | 5001.0     | 5187.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5092.5     | 3772.222 | 3.772  | 3962.0     | 5121.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5116.9     | 3790.321 | 3.790  | 3403.0     | 5178.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5114.3     | 3788.395 | 3.788  | 4961.0     | 5098.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5250.0     | 3888.914 | 3.889  | 5041.0     | 5172.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5130.0     | 3800.000 | 3.800  | 4897.0     | 5127.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5161.9     | 3823.605 | 3.824  | 5000.0     | 5161.0        |

Final op time: 3.889 us (5250.0 cycles), device=5, bucket=NCRISC transport

