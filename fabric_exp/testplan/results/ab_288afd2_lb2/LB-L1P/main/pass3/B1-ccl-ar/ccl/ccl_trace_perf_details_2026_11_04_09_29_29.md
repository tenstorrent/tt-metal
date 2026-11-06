# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_29_29
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_04_09_29_40/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1940.1     | 1437.111 | 1.437  | 1826.0     | 1946.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 1734.6     | 1284.914 | 1.285  | 1687.0     | 1719.5        |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1807.3     | 1338.765 | 1.339  | 1362.0     | 1827.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1891.1     | 1400.840 | 1.401  | 1461.0     | 1913.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 579.5      | 429.259  | 0.429  | 451.0      | 581.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 368.4      | 272.914  | 0.273  | 364.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1807.2     | 1338.667 | 1.339  | 1713.0     | 1796.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 1790.0     | 1325.901 | 1.326  | 1709.0     | 1788.0        |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1860.3     | 1377.975 | 1.378  | 1575.0     | 1835.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1952.3     | 1446.173 | 1.446  | 1667.0     | 1938.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 576.1      | 426.765  | 0.427  | 435.0      | 577.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.9      | 273.235  | 0.273  | 363.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1873.7     | 1387.889 | 1.388  | 1807.2                | 1873.7                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 1762.3     | 1305.407 | 1.305  | 1734.6                | 1762.3                   |
| NCRISC              | CCL_RECEIVER      | 2            | 1833.8     | 1358.370 | 1.358  | 1807.3                | 1833.8                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1921.7     | 1423.506 | 1.424  | 1891.1                | 1921.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 577.8      | 428.012  | 0.428  | 576.1                 | 577.8                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 368.6      | 273.074  | 0.273  | 368.4                 | 368.6                    |

Final op time: 1.424 us (1921.7 cycles), bucket=TRISC_0 compute

