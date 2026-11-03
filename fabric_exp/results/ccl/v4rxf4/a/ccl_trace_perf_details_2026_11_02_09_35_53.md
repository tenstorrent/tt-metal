# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_35_53
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_02_09_36_06/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1756.4     | 1301.037 | 1.301  | 1706.0     | 1750.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 1743.4     | 1291.407 | 1.291  | 1663.0     | 1733.5        |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1721.9     | 1275.506 | 1.276  | 1522.0     | 1719.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1809.0     | 1340.000 | 1.340  | 1616.0     | 1812.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 576.3      | 426.914  | 0.427  | 419.0      | 581.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.0      | 277.062  | 0.277  | 368.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1839.2     | 1362.370 | 1.362  | 1804.0     | 1834.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 1721.4     | 1275.136 | 1.275  | 1660.0     | 1717.0        |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1719.9     | 1273.975 | 1.274  | 1525.0     | 1709.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1813.0     | 1342.963 | 1.343  | 1610.0     | 1802.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 559.0      | 414.099  | 0.414  | 413.0      | 563.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.3      | 278.000  | 0.278  | 370.0      | 374.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1797.8     | 1331.704 | 1.332  | 1756.4                | 1797.8                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 1732.4     | 1283.272 | 1.283  | 1721.4                | 1732.4                   |
| NCRISC              | CCL_RECEIVER      | 2            | 1720.9     | 1274.741 | 1.275  | 1719.9                | 1720.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1811.0     | 1341.481 | 1.341  | 1809.0                | 1811.0                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 567.7      | 420.506  | 0.421  | 559.0                 | 567.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.7      | 277.531  | 0.278  | 374.0                 | 374.7                    |

Final op time: 1.341 us (1811.0 cycles), bucket=TRISC_0 compute

