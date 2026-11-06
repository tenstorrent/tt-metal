# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_33_39
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

Report: generated/profiler/reports/2026_11_04_09_33_53/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1650.3     | 1222.420 | 1.222  | 1592.0     | 1627.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 25.9       | 19.210   | 0.019  | 18.0       | 21.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1613.7     | 1195.358 | 1.195  | 1194.0     | 1624.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1705.8     | 1263.556 | 1.264  | 1290.0     | 1709.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 573.1      | 424.543  | 0.425  | 442.0      | 573.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.531  | 0.274  | 363.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1723.6     | 1276.765 | 1.277  | 1687.0     | 1723.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.9       | 14.000   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1618.1     | 1198.593 | 1.199  | 1465.0     | 1589.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1710.6     | 1267.111 | 1.267  | 1554.0     | 1683.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 569.6      | 421.901  | 0.422  | 406.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.5      | 273.704  | 0.274  | 364.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1687.0     | 1249.593 | 1.250  | 1650.3                | 1687.0                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.4       | 16.605   | 0.017  | 18.9                  | 22.4                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1615.9     | 1196.975 | 1.197  | 1613.7                | 1615.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1708.2     | 1265.333 | 1.265  | 1705.8                | 1708.2                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.4      | 423.222  | 0.423  | 569.6                 | 571.4                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.4      | 273.617  | 0.274  | 369.3                 | 369.4                    |

Final op time: 1.265 us (1708.2 cycles), bucket=TRISC_0 compute

