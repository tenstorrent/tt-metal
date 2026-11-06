# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_33_35
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_04_05_33_44/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1642.9     | 1216.938 | 1.217  | 1591.0     | 1626.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 27.8       | 20.568   | 0.021  | 18.0       | 23.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1606.7     | 1190.148 | 1.190  | 1396.0     | 1602.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1738.7     | 1287.951 | 1.288  | 1518.0     | 1738.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 572.8      | 424.321  | 0.424  | 445.0      | 575.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.5      | 273.704  | 0.274  | 364.0      | 369.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1737.3     | 1286.864 | 1.287  | 1668.0     | 1726.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.9       | 14.025   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1595.1     | 1181.556 | 1.182  | 1454.0     | 1563.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1728.6     | 1280.444 | 1.280  | 1590.0     | 1689.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 571.5      | 423.358  | 0.423  | 447.0      | 572.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.1      | 273.407  | 0.273  | 364.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1690.1     | 1251.901 | 1.252  | 1642.9                | 1690.1                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.4       | 17.296   | 0.017  | 18.9                  | 23.4                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1600.9     | 1185.852 | 1.186  | 1595.1                | 1600.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1733.7     | 1284.198 | 1.284  | 1728.6                | 1733.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.2      | 423.840  | 0.424  | 571.5                 | 572.2                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.3      | 273.556  | 0.274  | 369.1                 | 369.3                    |

Final op time: 1.284 us (1733.7 cycles), bucket=TRISC_0 compute

