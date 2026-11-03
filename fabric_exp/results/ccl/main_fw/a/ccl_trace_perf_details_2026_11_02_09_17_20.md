# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_17_20
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

Report: generated/profiler/reports/2026_11_02_09_17_30/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 77.0       | 57.062   | 0.057  | 72.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10725.9    | 7945.111 | 7.945  | 10538.0    | 10715.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10762.4    | 7972.173 | 7.972  | 9915.0     | 10784.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10623.7    | 7869.407 | 7.869  | 7116.0     | 10719.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10604.2    | 7854.963 | 7.855  | 4865.0     | 10791.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10686.2    | 7915.728 | 7.916  | 9680.0     | 10693.5       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10774.1    | 7980.790 | 7.981  | 10306.0    | 10788.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10634.6    | 7877.481 | 7.877  | 7525.0     | 10797.0       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10627.1    | 7871.901 | 7.872  | 5763.0     | 10848.0       |

Final op time: 7.981 us (10774.1 cycles), device=5, bucket=NCRISC transport

