# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_23_00_59
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

Report: generated/profiler/reports/2026_11_03_23_01_11/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 77.1       | 57.086   | 0.057  | 71.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10753.0    | 7965.185 | 7.965  | 10494.0    | 10741.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10806.6    | 8004.914 | 8.005  | 10485.0    | 10768.0       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10578.6    | 7836.000 | 7.836  | 5726.0     | 10769.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10534.7    | 7803.506 | 7.804  | 3016.0     | 10816.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10641.4    | 7882.543 | 7.883  | 8457.0     | 10707.5       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10789.2    | 7992.025 | 7.992  | 10601.0    | 10808.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10569.1    | 7828.938 | 7.829  | 5558.0     | 10770.0       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10526.4    | 7797.333 | 7.797  | 2827.0     | 10799.0       |

Final op time: 8.005 us (10806.6 cycles), device=1, bucket=NCRISC transport

