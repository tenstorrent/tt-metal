# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_50_50
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_03_22_51_01/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1637.2     | 1212.765 | 1.213  | 1586.0     | 1629.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 25.1       | 18.617   | 0.019  | 18.0       | 20.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1624.9     | 1203.605 | 1.204  | 1380.0     | 1597.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1716.0     | 1271.111 | 1.271  | 1475.0     | 1688.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 573.4      | 424.765  | 0.425  | 445.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.7      | 273.877  | 0.274  | 366.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1713.8     | 1269.457 | 1.269  | 1661.0     | 1686.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.4       | 14.346   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1608.9     | 1191.778 | 1.192  | 1431.0     | 1587.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1702.2     | 1260.914 | 1.261  | 1518.0     | 1684.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 567.0      | 420.000  | 0.420  | 405.0      | 572.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.0      | 273.333  | 0.273  | 364.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1675.5     | 1241.111 | 1.241  | 1637.2                | 1675.5                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.2       | 16.481   | 0.016  | 19.4                  | 22.2                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1616.9     | 1197.691 | 1.198  | 1608.9                | 1616.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1709.1     | 1266.012 | 1.266  | 1702.2                | 1709.1                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 570.2      | 422.383  | 0.422  | 567.0                 | 570.2                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.4      | 273.605  | 0.274  | 369.0                 | 369.4                    |

Final op time: 1.266 us (1709.1 cycles), bucket=TRISC_0 compute

