# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_28_07
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

Report: generated/profiler/reports/2026_11_04_05_28_22/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1649.7     | 1221.975 | 1.222  | 1597.0     | 1648.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 28.0       | 20.716   | 0.021  | 18.0       | 25.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1614.1     | 1195.630 | 1.196  | 1193.0     | 1613.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1747.2     | 1294.222 | 1.294  | 1311.0     | 1747.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 572.8      | 424.321  | 0.424  | 431.0      | 574.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.4      | 273.630  | 0.274  | 363.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1752.7     | 1298.296 | 1.298  | 1656.0     | 1733.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.4       | 15.086   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1607.5     | 1190.741 | 1.191  | 1418.0     | 1588.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1738.4     | 1287.679 | 1.288  | 1543.0     | 1721.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 573.8      | 425.012  | 0.425  | 441.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.556  | 0.274  | 356.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1701.2     | 1260.136 | 1.260  | 1649.7                | 1701.2                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.2       | 17.901   | 0.018  | 20.4                  | 24.2                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1610.8     | 1193.185 | 1.193  | 1607.5                | 1610.8                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1742.8     | 1290.951 | 1.291  | 1738.4                | 1742.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 573.3      | 424.667  | 0.425  | 572.8                 | 573.3                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.4      | 273.593  | 0.274  | 369.3                 | 369.4                    |

Final op time: 1.291 us (1742.8 cycles), bucket=TRISC_0 compute

