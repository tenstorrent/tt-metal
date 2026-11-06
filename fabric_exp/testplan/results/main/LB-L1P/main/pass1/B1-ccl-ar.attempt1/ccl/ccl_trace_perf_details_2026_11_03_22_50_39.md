# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_50_39
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

Report: generated/profiler/reports/2026_11_03_22_50_49/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1803.2     | 1335.728 | 1.336  | 1716.0     | 1801.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 1785.7     | 1322.741 | 1.323  | 1704.0     | 1781.0        |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1812.0     | 1342.222 | 1.342  | 1319.0     | 1836.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1904.8     | 1410.988 | 1.411  | 1406.0     | 1930.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 575.1      | 425.975  | 0.426  | 458.0      | 577.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.7      | 273.852  | 0.274  | 359.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1945.0     | 1440.741 | 1.441  | 1869.0     | 1946.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 1821.4     | 1349.160 | 1.349  | 1637.0     | 1843.5        |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1826.6     | 1353.012 | 1.353  | 1571.0     | 1779.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1916.1     | 1419.333 | 1.419  | 1657.0     | 1872.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 571.8      | 423.580  | 0.424  | 451.0      | 573.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.2      | 273.506  | 0.274  | 365.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1874.1     | 1388.235 | 1.388  | 1803.2                | 1874.1                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 1803.5     | 1335.951 | 1.336  | 1785.7                | 1803.5                   |
| NCRISC              | CCL_RECEIVER      | 2            | 1819.3     | 1347.617 | 1.348  | 1812.0                | 1819.3                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1910.5     | 1415.160 | 1.415  | 1904.8                | 1910.5                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 573.5      | 424.778  | 0.425  | 571.8                 | 573.5                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.5      | 273.679  | 0.274  | 369.2                 | 369.5                    |

Final op time: 1.415 us (1910.5 cycles), bucket=TRISC_0 compute

