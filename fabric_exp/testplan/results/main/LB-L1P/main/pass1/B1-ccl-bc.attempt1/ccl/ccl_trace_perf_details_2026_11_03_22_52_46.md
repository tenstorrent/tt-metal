# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_52_46
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

Report: generated/profiler/reports/2026_11_03_22_52_58/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 79.1       | 58.617   | 0.059  | 71.0       | 81.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10728.7    | 7947.160 | 7.947  | 10606.0    | 10716.0       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10713.9    | 7936.198 | 7.936  | 9179.0     | 10778.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10625.9    | 7871.037 | 7.871  | 7498.0     | 10693.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10626.1    | 7871.210 | 7.871  | 6283.0     | 10747.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10672.0    | 7905.160 | 7.905  | 9935.0     | 10684.0       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10753.3    | 7965.383 | 7.965  | 10175.0    | 10758.5       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10644.8    | 7885.012 | 7.885  | 8283.0     | 10706.5       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10626.3    | 7871.309 | 7.871  | 6437.0     | 10737.5       |

Final op time: 7.965 us (10753.3 cycles), device=5, bucket=NCRISC transport

