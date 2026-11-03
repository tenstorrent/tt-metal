# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_02_41
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

Report: generated/profiler/reports/2026_11_02_10_02_51/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.4       | 56.617   | 0.057  | 70.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10657.6    | 7894.519 | 7.895  | 10460.0    | 10695.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10680.7    | 7911.605 | 7.912  | 9597.0     | 10742.0       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10576.3    | 7834.272 | 7.834  | 7874.0     | 10671.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10550.3    | 7815.037 | 7.815  | 5786.0     | 10713.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10622.8    | 7868.741 | 7.869  | 9828.0     | 10644.0       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10690.4    | 7918.790 | 7.919  | 9884.0     | 10699.5       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10571.8    | 7830.988 | 7.831  | 7926.0     | 10666.0       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10564.4    | 7825.481 | 7.825  | 6291.0     | 10722.5       |

Final op time: 7.919 us (10690.4 cycles), device=5, bucket=NCRISC transport

