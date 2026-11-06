# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_33_16
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

Report: generated/profiler/reports/2026_11_04_09_33_34/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1997.8     | 1479.852 | 1.480  | 1953.0     | 1992.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.2       | 14.198   | 0.014  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1836.4     | 1360.272 | 1.360  | 1709.0     | 1837.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1970.0     | 1459.235 | 1.459  | 1844.0     | 1970.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 570.0      | 422.247  | 0.422  | 448.0      | 572.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.1      | 273.407  | 0.273  | 364.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1907.4     | 1412.889 | 1.413  | 1848.0     | 1917.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 32.8       | 24.296   | 0.024  | 18.0       | 33.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1886.8     | 1397.654 | 1.398  | 1775.0     | 1873.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 2014.0     | 1491.877 | 1.492  | 1904.0     | 2004.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 569.1      | 421.580  | 0.422  | 415.0      | 572.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.8      | 273.926  | 0.274  | 365.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1952.6     | 1446.370 | 1.446  | 1907.4                | 1952.6                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 26.0       | 19.247   | 0.019  | 19.2                  | 26.0                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1861.6     | 1378.963 | 1.379  | 1836.4                | 1861.6                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1992.0     | 1475.556 | 1.476  | 1970.0                | 1992.0                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 569.6      | 421.914  | 0.422  | 569.1                 | 569.6                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.5      | 273.667  | 0.274  | 369.1                 | 369.5                    |

Final op time: 1.476 us (1992.0 cycles), bucket=TRISC_0 compute

