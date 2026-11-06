# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_49_55
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_03_22_50_09/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1660.2     | 1229.753 | 1.230  | 1607.0     | 1665.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 31.0       | 22.963   | 0.023  | 18.0       | 30.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1650.2     | 1222.370 | 1.222  | 1300.0     | 1689.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1779.0     | 1317.753 | 1.318  | 1417.0     | 1822.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 576.2      | 426.790  | 0.427  | 455.0      | 576.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.0      | 273.309  | 0.273  | 360.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1791.6     | 1327.136 | 1.327  | 1665.0     | 1829.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.2       | 14.247   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1635.8     | 1211.704 | 1.212  | 1406.0     | 1589.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1767.8     | 1309.481 | 1.309  | 1519.0     | 1726.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 576.8      | 427.235  | 0.427  | 457.0      | 577.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.5      | 273.704  | 0.274  | 366.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1725.9     | 1278.444 | 1.278  | 1660.2                | 1725.9                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 25.1       | 18.605   | 0.019  | 19.2                  | 25.1                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1643.0     | 1217.037 | 1.217  | 1635.8                | 1643.0                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1773.4     | 1313.617 | 1.314  | 1767.8                | 1773.4                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 576.5      | 427.012  | 0.427  | 576.2                 | 576.5                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.2      | 273.506  | 0.274  | 369.0                 | 369.2                    |

Final op time: 1.314 us (1773.4 cycles), bucket=TRISC_0 compute

