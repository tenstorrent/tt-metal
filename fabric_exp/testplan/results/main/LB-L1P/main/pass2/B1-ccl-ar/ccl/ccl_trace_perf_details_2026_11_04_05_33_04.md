# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_33_04
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_04_05_33_14/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1647.7     | 1220.494 | 1.220  | 1602.0     | 1623.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 25.8       | 19.086   | 0.019  | 18.0       | 22.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1595.0     | 1181.481 | 1.181  | 1461.0     | 1566.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1728.8     | 1280.568 | 1.281  | 1601.0     | 1697.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.9      | 423.630  | 0.424  | 433.0      | 574.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.7      | 273.827  | 0.274  | 363.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1715.0     | 1270.346 | 1.270  | 1667.0     | 1684.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.0       | 14.049   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1579.4     | 1169.926 | 1.170  | 1419.0     | 1556.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1708.6     | 1265.605 | 1.266  | 1551.0     | 1688.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.8      | 424.321  | 0.424  | 431.0      | 575.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.0      | 273.358  | 0.273  | 365.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1681.3     | 1245.420 | 1.245  | 1647.7                | 1681.3                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.4       | 16.568   | 0.017  | 19.0                  | 22.4                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1587.2     | 1175.704 | 1.176  | 1579.4                | 1587.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1718.7     | 1273.086 | 1.273  | 1708.6                | 1718.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.4      | 423.975  | 0.424  | 571.9                 | 572.4                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.4      | 273.593  | 0.274  | 369.0                 | 369.4                    |

Final op time: 1.273 us (1718.7 cycles), bucket=TRISC_0 compute

