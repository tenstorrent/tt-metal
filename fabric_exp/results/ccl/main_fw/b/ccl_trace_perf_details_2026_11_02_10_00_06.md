# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_00_06
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_02_10_00_16/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1631.8     | 1208.765 | 1.209  | 1573.0     | 1641.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 27.2       | 20.123   | 0.020  | 18.0       | 22.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1563.2     | 1157.901 | 1.158  | 1394.0     | 1550.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1695.3     | 1255.778 | 1.256  | 1530.0     | 1682.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 567.5      | 420.346  | 0.420  | 393.0      | 573.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.3      | 277.235  | 0.277  | 370.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1696.1     | 1256.395 | 1.256  | 1649.0     | 1682.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.8       | 14.667   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1558.3     | 1154.272 | 1.154  | 1337.0     | 1547.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1694.2     | 1254.963 | 1.255  | 1470.0     | 1686.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 557.9      | 413.259  | 0.413  | 409.0      | 564.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.7      | 277.531  | 0.278  | 369.0      | 374.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1664.0     | 1232.580 | 1.233  | 1631.8                | 1664.0                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.5       | 17.395   | 0.017  | 19.8                  | 23.5                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1560.7     | 1156.086 | 1.156  | 1558.3                | 1560.7                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1694.8     | 1255.370 | 1.255  | 1694.2                | 1694.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 562.7      | 416.802  | 0.417  | 557.9                 | 562.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.5      | 277.383  | 0.277  | 374.3                 | 374.5                    |

Final op time: 1.255 us (1694.8 cycles), bucket=TRISC_0 compute

