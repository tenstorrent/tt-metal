# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_29_52
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_04_09_30_02/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1839.3     | 1362.469 | 1.362  | 1785.0     | 1841.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.1       | 14.173   | 0.014  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1703.3     | 1261.679 | 1.262  | 1432.0     | 1724.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1796.4     | 1330.667 | 1.331  | 1520.0     | 1816.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 569.6      | 421.951  | 0.422  | 426.0      | 569.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.2      | 273.457  | 0.273  | 363.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1698.9     | 1258.420 | 1.258  | 1618.0     | 1702.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 35.4       | 26.247   | 0.026  | 21.0       | 32.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1780.2     | 1318.642 | 1.319  | 1557.0     | 1759.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1874.9     | 1388.815 | 1.389  | 1643.0     | 1852.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.7      | 424.247  | 0.424  | 426.0      | 575.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 370.2      | 274.198  | 0.274  | 366.0      | 370.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1769.1     | 1310.444 | 1.310  | 1698.9                | 1769.1                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 27.3       | 20.210   | 0.020  | 19.1                  | 27.3                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1741.7     | 1290.160 | 1.290  | 1703.3                | 1741.7                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1835.7     | 1359.741 | 1.360  | 1796.4                | 1835.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.2      | 423.099  | 0.423  | 569.6                 | 571.2                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.7      | 273.827  | 0.274  | 369.2                 | 369.7                    |

Final op time: 1.360 us (1835.7 cycles), bucket=TRISC_0 compute

