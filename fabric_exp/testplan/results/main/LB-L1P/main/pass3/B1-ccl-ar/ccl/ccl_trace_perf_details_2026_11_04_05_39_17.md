# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_39_17
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

Report: generated/profiler/reports/2026_11_04_05_39_27/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1649.0     | 1221.457 | 1.221  | 1591.0     | 1640.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 27.3       | 20.198   | 0.020  | 18.0       | 26.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1630.2     | 1207.531 | 1.208  | 1477.0     | 1607.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1726.0     | 1278.519 | 1.279  | 1568.0     | 1702.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 567.2      | 420.123  | 0.420  | 425.0      | 572.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.8      | 273.951  | 0.274  | 365.0      | 370.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1725.3     | 1278.025 | 1.278  | 1670.0     | 1693.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.5       | 13.728   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1616.8     | 1197.605 | 1.198  | 1366.0     | 1576.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1708.6     | 1265.654 | 1.266  | 1471.0     | 1666.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 570.2      | 422.346  | 0.422  | 439.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.9      | 274.025  | 0.274  | 364.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1687.2     | 1249.741 | 1.250  | 1649.0                | 1687.2                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.9       | 16.963   | 0.017  | 18.5                  | 22.9                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1623.5     | 1202.568 | 1.203  | 1616.8                | 1623.5                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1717.3     | 1272.086 | 1.272  | 1708.6                | 1717.3                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 568.7      | 421.235  | 0.421  | 567.2                 | 568.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.9      | 273.988  | 0.274  | 369.8                 | 369.9                    |

Final op time: 1.272 us (1717.3 cycles), bucket=TRISC_0 compute

