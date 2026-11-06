# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_32_36
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

Report: generated/profiler/reports/2026_11_04_09_32_52/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1649.6     | 1221.926 | 1.222  | 1594.0     | 1646.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.3       | 19.481   | 0.019  | 18.0       | 25.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1591.9     | 1179.185 | 1.179  | 1415.0     | 1593.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1728.5     | 1280.395 | 1.280  | 1550.0     | 1722.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.1      | 423.012  | 0.423  | 445.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.6      | 273.778  | 0.274  | 363.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1722.9     | 1276.222 | 1.276  | 1677.0     | 1718.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.5       | 13.728   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1583.0     | 1172.617 | 1.173  | 1422.0     | 1558.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1719.2     | 1273.506 | 1.274  | 1544.0     | 1697.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 571.1      | 423.037  | 0.423  | 434.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.9      | 273.975  | 0.274  | 362.0      | 369.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1686.2     | 1249.074 | 1.249  | 1649.6                | 1686.2                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.4       | 16.605   | 0.017  | 18.5                  | 22.4                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1587.5     | 1175.901 | 1.176  | 1583.0                | 1587.5                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1723.9     | 1276.951 | 1.277  | 1719.2                | 1723.9                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.1      | 423.025  | 0.423  | 571.1                 | 571.1                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.7      | 273.877  | 0.274  | 369.6                 | 369.7                    |

Final op time: 1.277 us (1723.9 cycles), bucket=TRISC_0 compute

