# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_50_25
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

Report: generated/profiler/reports/2026_11_03_22_50_39/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1643.0     | 1217.037 | 1.217  | 1584.0     | 1626.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.4       | 19.556   | 0.020  | 18.0       | 21.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1639.7     | 1214.593 | 1.215  | 1249.0     | 1627.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1771.7     | 1312.370 | 1.312  | 1377.0     | 1769.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 574.5      | 425.531  | 0.426  | 452.0      | 576.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.6      | 273.753  | 0.274  | 363.0      | 369.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1776.4     | 1315.877 | 1.316  | 1681.0     | 1747.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.8       | 13.926   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1623.3     | 1202.420 | 1.202  | 1361.0     | 1573.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1754.8     | 1299.827 | 1.300  | 1495.0     | 1717.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 574.5      | 425.580  | 0.426  | 415.0      | 584.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.0      | 273.333  | 0.273  | 365.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1709.7     | 1266.457 | 1.266  | 1643.0                | 1709.7                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.6       | 16.741   | 0.017  | 18.8                  | 22.6                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1631.5     | 1208.506 | 1.209  | 1623.3                | 1631.5                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1763.2     | 1306.099 | 1.306  | 1754.8                | 1763.2                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 574.5      | 425.556  | 0.426  | 574.5                 | 574.5                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.3      | 273.543  | 0.274  | 369.0                 | 369.3                    |

Final op time: 1.306 us (1763.2 cycles), bucket=TRISC_0 compute

