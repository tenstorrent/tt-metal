# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_34_06
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_04_09_34_19/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1640.9     | 1215.481 | 1.215  | 1604.0     | 1625.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 27.3       | 20.222   | 0.020  | 18.0       | 21.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1620.8     | 1200.593 | 1.201  | 1503.0     | 1625.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1712.4     | 1268.420 | 1.268  | 1600.0     | 1719.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 574.2      | 425.309  | 0.425  | 405.0      | 576.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.6      | 273.778  | 0.274  | 366.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1715.9     | 1271.037 | 1.271  | 1686.0     | 1709.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.5       | 14.444   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1609.1     | 1191.951 | 1.192  | 1491.0     | 1577.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1702.6     | 1261.160 | 1.261  | 1576.0     | 1674.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 570.0      | 422.222  | 0.422  | 405.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.8      | 273.185  | 0.273  | 364.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1678.4     | 1243.259 | 1.243  | 1640.9                | 1678.4                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.4       | 17.333   | 0.017  | 19.5                  | 23.4                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1615.0     | 1196.272 | 1.196  | 1609.1                | 1615.0                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1707.5     | 1264.790 | 1.265  | 1702.6                | 1707.5                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.1      | 423.765  | 0.424  | 570.0                 | 572.1                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.2      | 273.481  | 0.273  | 368.8                 | 369.2                    |

Final op time: 1.265 us (1707.5 cycles), bucket=TRISC_0 compute

