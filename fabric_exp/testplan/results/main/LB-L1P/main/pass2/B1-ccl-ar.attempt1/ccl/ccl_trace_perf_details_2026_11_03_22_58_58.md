# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_58_58
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

Report: generated/profiler/reports/2026_11_03_22_59_11/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1649.2     | 1221.630 | 1.222  | 1594.0     | 1630.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.3       | 19.481   | 0.019  | 18.0       | 22.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1660.2     | 1229.802 | 1.230  | 1430.0     | 1653.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1750.9     | 1296.963 | 1.297  | 1499.0     | 1742.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 573.5      | 424.815  | 0.425  | 439.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 370.2      | 274.222  | 0.274  | 366.0      | 370.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1759.1     | 1303.012 | 1.303  | 1669.0     | 1725.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.0       | 14.074   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1648.6     | 1221.160 | 1.221  | 1433.0     | 1626.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1738.1     | 1287.457 | 1.287  | 1525.0     | 1710.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 573.1      | 424.519  | 0.425  | 470.0      | 575.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.9      | 273.284  | 0.273  | 363.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1704.1     | 1262.321 | 1.262  | 1649.2                | 1704.1                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.6       | 16.778   | 0.017  | 19.0                  | 22.6                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1654.4     | 1225.481 | 1.225  | 1648.6                | 1654.4                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1744.5     | 1292.210 | 1.292  | 1738.1                | 1744.5                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 573.3      | 424.667  | 0.425  | 573.1                 | 573.3                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.6      | 273.753  | 0.274  | 368.9                 | 369.6                    |

Final op time: 1.292 us (1744.5 cycles), bucket=TRISC_0 compute

