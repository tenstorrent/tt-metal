# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_51_12
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

Report: generated/profiler/reports/2026_11_03_22_51_21/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1650.3     | 1222.420 | 1.222  | 1590.0     | 1629.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.8       | 19.827   | 0.020  | 18.0       | 20.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1659.1     | 1228.938 | 1.229  | 1487.0     | 1651.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1751.5     | 1297.383 | 1.297  | 1566.0     | 1746.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 574.5      | 425.556  | 0.426  | 465.0      | 574.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.1      | 273.432  | 0.273  | 363.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1757.7     | 1301.975 | 1.302  | 1697.0     | 1729.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.8       | 13.901   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1639.3     | 1214.296 | 1.214  | 1484.0     | 1620.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1733.0     | 1283.704 | 1.284  | 1570.0     | 1718.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 569.7      | 422.025  | 0.422  | 425.0      | 573.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.8      | 273.926  | 0.274  | 366.0      | 370.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1704.0     | 1262.198 | 1.262  | 1650.3                | 1704.0                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.8       | 16.864   | 0.017  | 18.8                  | 22.8                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1649.2     | 1221.617 | 1.222  | 1639.3                | 1649.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1742.2     | 1290.543 | 1.291  | 1733.0                | 1742.2                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.1      | 423.790  | 0.424  | 569.7                 | 572.1                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.5      | 273.679  | 0.274  | 369.1                 | 369.5                    |

Final op time: 1.291 us (1742.2 cycles), bucket=TRISC_0 compute

