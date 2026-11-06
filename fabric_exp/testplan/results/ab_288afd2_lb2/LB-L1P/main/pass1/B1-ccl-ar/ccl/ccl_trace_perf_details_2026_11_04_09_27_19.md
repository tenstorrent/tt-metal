# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_27_19
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

Report: generated/profiler/reports/2026_11_04_09_27_36/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1907.0     | 1412.593 | 1.413  | 1850.0     | 1911.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 28.7       | 21.284   | 0.021  | 18.0       | 28.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1875.4     | 1389.210 | 1.389  | 1667.0     | 1891.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 2006.2     | 1486.074 | 1.486  | 1794.0     | 2025.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 569.7      | 421.975  | 0.422  | 476.0      | 572.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.531  | 0.274  | 361.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 2007.9     | 1487.358 | 1.487  | 1901.0     | 2014.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.3       | 14.296   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1857.7     | 1376.099 | 1.376  | 1635.0     | 1855.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1988.7     | 1473.086 | 1.473  | 1768.0     | 1984.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 569.7      | 422.025  | 0.422  | 446.0      | 569.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.2      | 273.481  | 0.273  | 365.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1957.5     | 1449.975 | 1.450  | 1907.0                | 1957.5                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.0       | 17.790   | 0.018  | 19.3                  | 24.0                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1866.6     | 1382.654 | 1.383  | 1857.7                | 1866.6                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1997.4     | 1479.580 | 1.480  | 1988.7                | 1997.4                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 569.7      | 422.000  | 0.422  | 569.7                 | 569.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.2      | 273.506  | 0.274  | 369.2                 | 369.2                    |

Final op time: 1.480 us (1997.4 cycles), bucket=TRISC_0 compute

