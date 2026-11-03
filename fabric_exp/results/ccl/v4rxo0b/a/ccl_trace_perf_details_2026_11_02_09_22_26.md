# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_22_26
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_02_09_22_39/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1622.5     | 1201.827 | 1.202  | 1575.0     | 1612.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 25.5       | 18.889   | 0.019  | 18.0       | 20.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1593.0     | 1180.025 | 1.180  | 1409.0     | 1599.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1689.3     | 1251.309 | 1.251  | 1513.0     | 1702.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.6      | 423.407  | 0.423  | 431.0      | 578.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.5      | 277.383  | 0.277  | 371.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1693.5     | 1254.420 | 1.254  | 1653.0     | 1690.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.7       | 15.309   | 0.015  | 18.0       | 20.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1584.4     | 1173.630 | 1.174  | 1472.0     | 1553.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1681.8     | 1245.753 | 1.246  | 1576.0     | 1653.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 568.8      | 421.333  | 0.421  | 455.0      | 574.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.2      | 277.926  | 0.278  | 368.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1658.0     | 1228.123 | 1.228  | 1622.5                | 1658.0                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.1       | 17.099   | 0.017  | 20.7                  | 23.1                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1588.7     | 1176.827 | 1.177  | 1584.4                | 1588.7                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1685.5     | 1248.531 | 1.249  | 1681.8                | 1685.5                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 570.2      | 422.370  | 0.422  | 568.8                 | 570.2                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.8      | 277.654  | 0.278  | 374.5                 | 374.8                    |

Final op time: 1.249 us (1685.5 cycles), bucket=TRISC_0 compute

