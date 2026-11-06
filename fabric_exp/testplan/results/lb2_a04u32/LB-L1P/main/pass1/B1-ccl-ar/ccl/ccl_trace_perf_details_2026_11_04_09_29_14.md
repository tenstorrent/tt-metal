# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_29_14
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

Report: generated/profiler/reports/2026_11_04_09_29_24/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1652.9     | 1224.395 | 1.224  | 1589.0     | 1667.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 24.8       | 18.395   | 0.018  | 18.0       | 22.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1658.6     | 1228.568 | 1.229  | 1350.0     | 1650.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1753.1     | 1298.568 | 1.299  | 1446.0     | 1744.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 570.8      | 422.815  | 0.423  | 442.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 370.2      | 274.222  | 0.274  | 364.0      | 370.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1755.7     | 1300.519 | 1.301  | 1658.0     | 1729.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.7       | 14.593   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1645.2     | 1218.667 | 1.219  | 1412.0     | 1618.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1739.1     | 1288.247 | 1.288  | 1512.0     | 1717.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 567.9      | 420.642  | 0.421  | 442.0      | 571.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.531  | 0.274  | 364.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1704.3     | 1262.457 | 1.262  | 1652.9                | 1704.3                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.3       | 16.494   | 0.016  | 19.7                  | 22.3                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1651.9     | 1223.617 | 1.224  | 1645.2                | 1651.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1746.1     | 1293.407 | 1.293  | 1739.1                | 1746.1                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 569.3      | 421.728  | 0.422  | 567.9                 | 569.3                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.7      | 273.877  | 0.274  | 369.3                 | 369.7                    |

Final op time: 1.293 us (1746.1 cycles), bucket=TRISC_0 compute

