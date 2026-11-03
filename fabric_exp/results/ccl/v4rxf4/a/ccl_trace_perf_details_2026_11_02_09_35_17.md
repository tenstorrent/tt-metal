# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_35_17
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_02_09_35_35/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1622.0     | 1201.457 | 1.201  | 1574.0     | 1609.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 25.0       | 18.543   | 0.019  | 18.0       | 22.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1564.0     | 1158.543 | 1.159  | 1456.0     | 1561.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1697.7     | 1257.531 | 1.258  | 1589.0     | 1696.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 566.9      | 419.926  | 0.420  | 466.0      | 573.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.5      | 277.407  | 0.277  | 368.0      | 375.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1696.3     | 1256.519 | 1.257  | 1652.0     | 1693.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.5       | 15.160   | 0.015  | 18.0       | 19.5          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1553.7     | 1150.864 | 1.151  | 1412.0     | 1537.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1690.0     | 1251.877 | 1.252  | 1552.0     | 1677.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 562.4      | 416.617  | 0.417  | 426.0      | 565.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.9      | 277.728  | 0.278  | 372.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1659.1     | 1228.988 | 1.229  | 1622.0                | 1659.1                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.8       | 16.852   | 0.017  | 20.5                  | 22.8                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1558.8     | 1154.704 | 1.155  | 1553.7                | 1558.8                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1693.8     | 1254.704 | 1.255  | 1690.0                | 1693.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 564.7      | 418.272  | 0.418  | 562.4                 | 564.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.7      | 277.568  | 0.278  | 374.5                 | 374.7                    |

Final op time: 1.255 us (1693.8 cycles), bucket=TRISC_0 compute

