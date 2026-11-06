# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_28_54
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

Report: generated/profiler/reports/2026_11_04_09_29_04/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1645.8     | 1219.136 | 1.219  | 1597.0     | 1623.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.4       | 19.531   | 0.020  | 18.0       | 21.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1623.7     | 1202.716 | 1.203  | 1217.0     | 1644.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1715.7     | 1270.914 | 1.271  | 1305.0     | 1737.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 572.5      | 424.049  | 0.424  | 474.0      | 572.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.556  | 0.274  | 365.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1728.7     | 1280.519 | 1.281  | 1653.0     | 1725.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.3       | 14.272   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1630.7     | 1207.901 | 1.208  | 1445.0     | 1620.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1723.3     | 1276.494 | 1.276  | 1540.0     | 1696.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 567.9      | 420.642  | 0.421  | 442.0      | 574.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.8      | 273.901  | 0.274  | 364.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1687.3     | 1249.827 | 1.250  | 1645.8                | 1687.3                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.8       | 16.901   | 0.017  | 19.3                  | 22.8                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1627.2     | 1205.309 | 1.205  | 1623.7                | 1627.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1719.5     | 1273.704 | 1.274  | 1715.7                | 1719.5                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 570.2      | 422.346  | 0.422  | 567.9                 | 570.2                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.5      | 273.728  | 0.274  | 369.3                 | 369.5                    |

Final op time: 1.274 us (1719.5 cycles), bucket=TRISC_0 compute

