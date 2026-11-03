# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_54_21
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

Report: generated/profiler/reports/2026_11_02_09_54_33/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1747.1     | 1294.173 | 1.294  | 1694.0     | 1745.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 1742.1     | 1290.469 | 1.290  | 1670.0     | 1748.0        |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1724.6     | 1277.506 | 1.278  | 1489.0     | 1724.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1811.3     | 1341.679 | 1.342  | 1596.0     | 1818.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 581.1      | 430.444  | 0.430  | 420.0      | 586.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 373.9      | 276.963  | 0.277  | 368.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1841.6     | 1364.173 | 1.364  | 1795.0     | 1840.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 1723.4     | 1276.617 | 1.277  | 1669.0     | 1725.0        |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1715.3     | 1270.617 | 1.271  | 1606.0     | 1697.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1809.9     | 1340.691 | 1.341  | 1683.0     | 1788.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 566.0      | 419.284  | 0.419  | 456.0      | 567.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.9      | 277.704  | 0.278  | 369.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1794.4     | 1329.173 | 1.329  | 1747.1                | 1794.4                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 1732.8     | 1283.543 | 1.284  | 1723.4                | 1732.8                   |
| NCRISC              | CCL_RECEIVER      | 2            | 1720.0     | 1274.062 | 1.274  | 1715.3                | 1720.0                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1810.6     | 1341.185 | 1.341  | 1809.9                | 1810.6                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 573.6      | 424.864  | 0.425  | 566.0                 | 573.6                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.4      | 277.333  | 0.277  | 373.9                 | 374.4                    |

Final op time: 1.341 us (1810.6 cycles), bucket=TRISC_0 compute

