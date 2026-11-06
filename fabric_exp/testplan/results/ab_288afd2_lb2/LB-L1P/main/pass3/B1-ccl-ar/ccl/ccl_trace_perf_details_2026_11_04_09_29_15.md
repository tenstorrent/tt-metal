# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_29_15
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

Report: generated/profiler/reports/2026_11_04_09_29_29/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1792.0     | 1327.432 | 1.327  | 1722.0     | 1816.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.8       | 13.951   | 0.014  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1625.0     | 1203.728 | 1.204  | 1280.0     | 1656.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1762.1     | 1305.284 | 1.305  | 1425.0     | 1792.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 570.9      | 422.864  | 0.423  | 438.0      | 574.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 368.7      | 273.111  | 0.273  | 364.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1674.1     | 1240.049 | 1.240  | 1617.0     | 1676.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 32.4       | 24.000   | 0.024  | 18.0       | 34.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1698.8     | 1258.370 | 1.258  | 1489.0     | 1693.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1833.5     | 1358.173 | 1.358  | 1614.0     | 1842.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 571.2      | 423.136  | 0.423  | 428.0      | 575.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.7      | 273.852  | 0.274  | 363.0      | 369.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1733.0     | 1283.741 | 1.284  | 1674.1                | 1733.0                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 25.6       | 18.975   | 0.019  | 18.8                  | 25.6                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1661.9     | 1231.049 | 1.231  | 1625.0                | 1661.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1797.8     | 1331.728 | 1.332  | 1762.1                | 1797.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.0      | 423.000  | 0.423  | 570.9                 | 571.0                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.2      | 273.481  | 0.273  | 368.7                 | 369.2                    |

Final op time: 1.332 us (1797.8 cycles), bucket=TRISC_0 compute

