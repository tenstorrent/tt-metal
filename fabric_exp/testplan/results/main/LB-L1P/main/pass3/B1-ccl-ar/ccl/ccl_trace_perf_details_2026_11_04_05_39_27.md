# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_39_27
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_04_05_39_37/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1657.3     | 1227.630 | 1.228  | 1591.0     | 1661.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 28.5       | 21.136   | 0.021  | 18.0       | 30.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1684.5     | 1247.802 | 1.248  | 1480.0     | 1711.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1779.4     | 1318.049 | 1.318  | 1572.0     | 1809.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 568.4      | 421.062  | 0.421  | 436.0      | 573.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 370.0      | 274.074  | 0.274  | 366.0      | 370.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1783.0     | 1320.716 | 1.321  | 1697.0     | 1751.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.6       | 13.778   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1660.3     | 1229.852 | 1.230  | 1416.0     | 1659.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1753.0     | 1298.519 | 1.299  | 1513.0     | 1754.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 571.0      | 422.963  | 0.423  | 459.0      | 572.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.4      | 273.654  | 0.274  | 363.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1720.1     | 1274.173 | 1.274  | 1657.3                | 1720.1                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.6       | 17.457   | 0.017  | 18.6                  | 23.6                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1672.4     | 1238.827 | 1.239  | 1660.3                | 1672.4                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1766.2     | 1308.284 | 1.308  | 1753.0                | 1766.2                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 569.7      | 422.012  | 0.422  | 568.4                 | 569.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.7      | 273.864  | 0.274  | 369.4                 | 369.7                    |

Final op time: 1.308 us (1766.2 cycles), bucket=TRISC_0 compute

