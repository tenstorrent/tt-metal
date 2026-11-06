# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_28_05
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

Report: generated/profiler/reports/2026_11_04_09_28_19/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1651.0     | 1222.938 | 1.223  | 1592.0     | 1639.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.1       | 19.333   | 0.019  | 18.0       | 22.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1605.3     | 1189.136 | 1.189  | 1381.0     | 1607.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1739.5     | 1288.543 | 1.289  | 1507.0     | 1740.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.8      | 423.556  | 0.424  | 460.0      | 574.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.7      | 273.877  | 0.274  | 366.0      | 370.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1747.9     | 1294.716 | 1.295  | 1658.0     | 1731.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.3       | 13.580   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1598.4     | 1184.025 | 1.184  | 1428.0     | 1591.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1735.0     | 1285.210 | 1.285  | 1576.0     | 1727.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 571.1      | 423.012  | 0.423  | 438.0      | 572.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.4      | 273.654  | 0.274  | 366.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1699.4     | 1258.827 | 1.259  | 1651.0                | 1699.4                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.2       | 16.457   | 0.016  | 18.3                  | 22.2                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1601.9     | 1186.580 | 1.187  | 1598.4                | 1601.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1737.3     | 1286.877 | 1.287  | 1735.0                | 1737.3                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.4      | 423.284  | 0.423  | 571.1                 | 571.4                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.6      | 273.765  | 0.274  | 369.4                 | 369.6                    |

Final op time: 1.287 us (1737.3 cycles), bucket=TRISC_0 compute

