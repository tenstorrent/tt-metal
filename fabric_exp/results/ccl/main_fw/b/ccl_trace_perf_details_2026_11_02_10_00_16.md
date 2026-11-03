# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_00_16
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_02_10_00_25/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1636.6     | 1212.296 | 1.212  | 1584.0     | 1644.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 25.4       | 18.840   | 0.019  | 18.0       | 24.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1625.7     | 1204.198 | 1.204  | 1174.0     | 1617.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1755.2     | 1300.173 | 1.300  | 1310.0     | 1742.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 566.5      | 419.630  | 0.420  | 405.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.0      | 277.062  | 0.277  | 370.0      | 373.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1769.4     | 1310.691 | 1.311  | 1653.0     | 1743.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 21.8       | 16.123   | 0.016  | 18.0       | 20.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1625.2     | 1203.827 | 1.204  | 1402.0     | 1599.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1762.8     | 1305.753 | 1.306  | 1539.0     | 1734.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 559.0      | 414.074  | 0.414  | 436.0      | 563.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.6      | 278.247  | 0.278  | 371.0      | 375.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1703.0     | 1261.494 | 1.261  | 1636.6                | 1703.0                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.6       | 17.481   | 0.017  | 21.8                  | 23.6                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1625.4     | 1204.012 | 1.204  | 1625.2                | 1625.4                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1759.0     | 1302.963 | 1.303  | 1755.2                | 1759.0                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 562.8      | 416.852  | 0.417  | 559.0                 | 562.8                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.8      | 277.654  | 0.278  | 374.0                 | 374.8                    |

Final op time: 1.303 us (1759.0 cycles), bucket=TRISC_0 compute

