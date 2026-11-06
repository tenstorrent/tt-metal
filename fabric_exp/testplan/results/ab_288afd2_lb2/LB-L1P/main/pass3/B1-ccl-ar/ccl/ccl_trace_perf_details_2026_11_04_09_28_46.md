# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_28_46
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

Report: generated/profiler/reports/2026_11_04_09_29_00/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1746.7     | 1293.852 | 1.294  | 1698.0     | 1728.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.4       | 14.395   | 0.014  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1594.8     | 1181.309 | 1.181  | 1282.0     | 1594.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1728.0     | 1280.000 | 1.280  | 1407.0     | 1725.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 575.6      | 426.370  | 0.426  | 411.0      | 579.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.4      | 273.654  | 0.274  | 356.0      | 369.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1680.4     | 1244.741 | 1.245  | 1628.0     | 1683.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 32.7       | 24.222   | 0.024  | 18.0       | 32.5          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1650.5     | 1222.593 | 1.223  | 1364.0     | 1603.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1786.9     | 1323.630 | 1.324  | 1493.0     | 1736.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.7      | 424.198  | 0.424  | 429.0      | 576.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.0      | 273.358  | 0.273  | 363.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1713.6     | 1269.296 | 1.269  | 1680.4                | 1713.6                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 26.1       | 19.309   | 0.019  | 19.4                  | 26.1                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1622.6     | 1201.951 | 1.202  | 1594.8                | 1622.6                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1757.5     | 1301.815 | 1.302  | 1728.0                | 1757.5                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 574.1      | 425.284  | 0.425  | 572.7                 | 574.1                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.2      | 273.506  | 0.274  | 369.0                 | 369.2                    |

Final op time: 1.302 us (1757.5 cycles), bucket=TRISC_0 compute

