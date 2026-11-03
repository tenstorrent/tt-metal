# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_38_23
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

Report: generated/profiler/reports/2026_11_02_09_38_37/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 77.0       | 57.012   | 0.057  | 70.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10565.9    | 7826.593 | 7.827  | 10448.0    | 10573.0       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10605.3    | 7855.802 | 7.856  | 9969.0     | 10621.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10453.4    | 7743.259 | 7.743  | 6839.0     | 10580.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10444.5    | 7736.667 | 7.737  | 5065.0     | 10626.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10515.6    | 7789.309 | 7.789  | 9193.0     | 10588.5       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10614.7    | 7862.716 | 7.863  | 10288.0    | 10612.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10467.2    | 7753.457 | 7.753  | 7391.0     | 10554.5       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10450.1    | 7740.815 | 7.741  | 5306.0     | 10652.5       |

Final op time: 7.863 us (10614.7 cycles), device=5, bucket=NCRISC transport

