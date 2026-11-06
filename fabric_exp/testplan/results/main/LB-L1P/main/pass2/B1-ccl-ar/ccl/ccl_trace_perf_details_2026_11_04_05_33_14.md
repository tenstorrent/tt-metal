# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_33_14
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

Report: generated/profiler/reports/2026_11_04_05_33_24/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1661.9     | 1231.062 | 1.231  | 1591.0     | 1677.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 28.9       | 21.383   | 0.021  | 18.0       | 29.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1656.5     | 1227.012 | 1.227  | 1445.0     | 1633.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1787.5     | 1324.049 | 1.324  | 1574.0     | 1752.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 574.6      | 425.654  | 0.426  | 413.0      | 577.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.7      | 273.877  | 0.274  | 366.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1774.6     | 1314.519 | 1.315  | 1655.0     | 1742.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.8       | 13.926   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1615.4     | 1196.617 | 1.197  | 1452.0     | 1599.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1754.0     | 1299.259 | 1.299  | 1593.0     | 1741.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 577.0      | 427.432  | 0.427  | 478.0      | 577.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.1      | 273.383  | 0.273  | 366.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1718.3     | 1272.790 | 1.273  | 1661.9                | 1718.3                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.8       | 17.654   | 0.018  | 18.8                  | 23.8                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1636.0     | 1211.815 | 1.212  | 1615.4                | 1636.0                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1770.7     | 1311.654 | 1.312  | 1754.0                | 1770.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 575.8      | 426.543  | 0.427  | 574.6                 | 575.8                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.4      | 273.630  | 0.274  | 369.1                 | 369.4                    |

Final op time: 1.312 us (1770.7 cycles), bucket=TRISC_0 compute

