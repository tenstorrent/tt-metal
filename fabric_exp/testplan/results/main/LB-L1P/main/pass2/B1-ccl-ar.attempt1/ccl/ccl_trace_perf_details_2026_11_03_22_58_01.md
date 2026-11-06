# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_58_01
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_03_22_58_14/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1637.4     | 1212.864 | 1.213  | 1592.0     | 1622.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 25.2       | 18.667   | 0.019  | 18.0       | 21.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1588.2     | 1176.469 | 1.176  | 1169.0     | 1599.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1723.0     | 1276.272 | 1.276  | 1308.0     | 1735.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 569.0      | 421.457  | 0.421  | 452.0      | 573.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.9      | 274.000  | 0.274  | 366.0      | 370.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1727.4     | 1279.580 | 1.280  | 1651.0     | 1722.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.5       | 14.444   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1602.1     | 1186.765 | 1.187  | 1349.0     | 1540.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1734.3     | 1284.667 | 1.285  | 1488.0     | 1676.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 575.9      | 426.617  | 0.427  | 446.0      | 577.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.580  | 0.274  | 366.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1682.4     | 1246.222 | 1.246  | 1637.4                | 1682.4                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.4       | 16.556   | 0.017  | 19.5                  | 22.4                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1595.2     | 1181.617 | 1.182  | 1588.2                | 1595.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1728.6     | 1280.469 | 1.280  | 1723.0                | 1728.6                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.5      | 424.037  | 0.424  | 569.0                 | 572.5                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.6      | 273.790  | 0.274  | 369.3                 | 369.6                    |

Final op time: 1.280 us (1728.6 cycles), bucket=TRISC_0 compute

