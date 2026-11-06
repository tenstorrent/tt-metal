# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_51_22
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_03_22_51_32/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1680.7     | 1244.963 | 1.245  | 1592.0     | 1691.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 23.7       | 17.531   | 0.018  | 18.0       | 20.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1679.3     | 1243.926 | 1.244  | 1308.0     | 1663.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1772.9     | 1313.259 | 1.313  | 1396.0     | 1767.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 572.6      | 424.148  | 0.424  | 437.0      | 574.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.6      | 273.778  | 0.274  | 364.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1776.1     | 1315.605 | 1.316  | 1688.0     | 1783.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.8       | 14.667   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1665.7     | 1233.852 | 1.234  | 1458.0     | 1643.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1760.7     | 1304.222 | 1.304  | 1551.0     | 1753.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 568.8      | 421.333  | 0.421  | 451.0      | 572.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.2      | 273.506  | 0.274  | 362.0      | 368.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1728.4     | 1280.284 | 1.280  | 1680.7                | 1728.4                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 21.7       | 16.099   | 0.016  | 19.8                  | 21.7                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1672.5     | 1238.889 | 1.239  | 1665.7                | 1672.5                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1766.8     | 1308.741 | 1.309  | 1760.7                | 1766.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 570.7      | 422.741  | 0.423  | 568.8                 | 570.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.4      | 273.642  | 0.274  | 369.2                 | 369.4                    |

Final op time: 1.309 us (1766.8 cycles), bucket=TRISC_0 compute

