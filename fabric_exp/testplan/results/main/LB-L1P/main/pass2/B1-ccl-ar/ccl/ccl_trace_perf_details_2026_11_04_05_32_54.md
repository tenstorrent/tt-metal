# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_32_54
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_04_05_33_04/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1899.7     | 1407.185 | 1.407  | 1849.0     | 1910.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 30.7       | 22.741   | 0.023  | 18.0       | 29.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1870.5     | 1385.531 | 1.386  | 1496.0     | 1884.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 2004.2     | 1484.617 | 1.485  | 1635.0     | 2017.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 569.6      | 421.901  | 0.422  | 447.0      | 573.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.2      | 273.506  | 0.274  | 366.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 2004.4     | 1484.741 | 1.485  | 1911.0     | 2007.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.8       | 13.926   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1853.3     | 1372.815 | 1.373  | 1656.0     | 1837.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1981.4     | 1467.728 | 1.468  | 1787.0     | 1951.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.2      | 423.877  | 0.424  | 463.0      | 573.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.6      | 273.012  | 0.273  | 363.0      | 368.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1952.1     | 1445.963 | 1.446  | 1899.7                | 1952.1                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.8       | 18.333   | 0.018  | 18.8                  | 24.8                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1861.9     | 1379.173 | 1.379  | 1853.3                | 1861.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1992.8     | 1476.173 | 1.476  | 1981.4                | 1992.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 570.9      | 422.889  | 0.423  | 569.6                 | 570.9                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 368.9      | 273.259  | 0.273  | 368.6                 | 368.9                    |

Final op time: 1.476 us (1992.8 cycles), bucket=TRISC_0 compute

