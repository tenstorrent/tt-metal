# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_54_34
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_02_09_54_47/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1616.3     | 1197.235 | 1.197  | 1573.0     | 1608.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 24.6       | 18.247   | 0.018  | 18.0       | 21.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1582.6     | 1172.296 | 1.172  | 1167.0     | 1590.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1674.7     | 1240.494 | 1.240  | 1266.0     | 1682.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 578.8      | 428.716  | 0.429  | 448.0      | 584.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.5      | 277.432  | 0.277  | 371.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1696.8     | 1256.914 | 1.257  | 1656.0     | 1690.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.8       | 14.691   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1586.1     | 1174.914 | 1.175  | 1449.0     | 1557.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1680.1     | 1244.494 | 1.244  | 1545.0     | 1642.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 570.3      | 422.469  | 0.422  | 440.0      | 582.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.3      | 277.259  | 0.277  | 369.0      | 374.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1656.5     | 1227.074 | 1.227  | 1616.3                | 1656.5                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.2       | 16.469   | 0.016  | 19.8                  | 22.2                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1584.4     | 1173.605 | 1.174  | 1582.6                | 1584.4                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1677.4     | 1242.494 | 1.242  | 1674.7                | 1677.4                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 574.5      | 425.593  | 0.426  | 570.3                 | 574.5                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.4      | 277.346  | 0.277  | 374.3                 | 374.4                    |

Final op time: 1.242 us (1677.4 cycles), bucket=TRISC_0 compute

