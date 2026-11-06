# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_41_08
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

Report: generated/profiler/reports/2026_11_04_05_41_18/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 79.4       | 58.815   | 0.059  | 72.0       | 81.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10778.7    | 7984.247 | 7.984  | 10583.0    | 10762.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10788.8    | 7991.679 | 7.992  | 9648.0     | 10804.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10699.3    | 7925.383 | 7.925  | 7955.0     | 10800.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10663.6    | 7898.988 | 7.899  | 5800.0     | 10847.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10731.4    | 7949.210 | 7.949  | 10117.0    | 10762.5       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10814.2    | 8010.519 | 8.011  | 10112.0    | 10856.5       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10712.3    | 7935.037 | 7.935  | 8159.0     | 10806.0       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10708.6    | 7932.272 | 7.932  | 6855.0     | 10858.5       |

Final op time: 8.011 us (10814.2 cycles), device=5, bucket=NCRISC transport

