# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_38_38
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

### num_links=2, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_02_09_38_51/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.0       | 56.272   | 0.056  | 70.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10614.8    | 7862.790 | 7.863  | 10532.0    | 10608.0       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10641.1    | 7882.321 | 7.882  | 9678.0     | 10672.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10517.1    | 7790.444 | 7.790  | 7571.0     | 10628.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10512.2    | 7786.815 | 7.787  | 5887.0     | 10660.5       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10571.8    | 7830.988 | 7.831  | 9674.0     | 10581.0       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10648.8    | 7887.975 | 7.888  | 10135.0    | 10644.5       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10527.6    | 7798.247 | 7.798  | 7751.0     | 10602.0       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10522.5    | 7794.469 | 7.794  | 6216.0     | 10655.0       |

Final op time: 7.888 us (10648.8 cycles), device=5, bucket=NCRISC transport

