# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_00_46
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

Report: generated/profiler/reports/2026_11_02_10_00_56/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1763.9     | 1306.617 | 1.307  | 1707.0     | 1750.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 1784.9     | 1322.123 | 1.322  | 1701.0     | 1786.5        |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1769.1     | 1310.469 | 1.310  | 1585.0     | 1770.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1861.0     | 1378.494 | 1.378  | 1660.0     | 1867.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 574.1      | 425.259  | 0.425  | 426.0      | 576.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.3      | 277.259  | 0.277  | 368.0      | 374.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1890.6     | 1400.444 | 1.400  | 1798.0     | 1854.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 1728.6     | 1280.444 | 1.280  | 1684.0     | 1725.0        |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1761.6     | 1304.889 | 1.305  | 1586.0     | 1726.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1854.7     | 1373.827 | 1.374  | 1655.0     | 1811.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 559.6      | 414.519  | 0.415  | 402.0      | 562.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.8      | 277.630  | 0.278  | 371.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1827.3     | 1353.531 | 1.354  | 1763.9                | 1827.3                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 1756.7     | 1301.284 | 1.301  | 1728.6                | 1756.7                   |
| NCRISC              | CCL_RECEIVER      | 2            | 1765.4     | 1307.679 | 1.308  | 1761.6                | 1765.4                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1857.8     | 1376.160 | 1.376  | 1854.7                | 1857.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 566.9      | 419.889  | 0.420  | 559.6                 | 566.9                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.6      | 277.444  | 0.277  | 374.3                 | 374.6                    |

Final op time: 1.376 us (1857.8 cycles), bucket=TRISC_0 compute

