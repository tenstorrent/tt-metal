# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_54_08
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

Report: generated/profiler/reports/2026_11_02_09_54_21/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1626.5     | 1204.840 | 1.205  | 1581.0     | 1616.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.5       | 19.630   | 0.020  | 18.0       | 22.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1574.4     | 1166.247 | 1.166  | 1299.0     | 1576.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1704.4     | 1262.543 | 1.263  | 1427.0     | 1707.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 568.2      | 420.889  | 0.421  | 416.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.4      | 277.309  | 0.277  | 370.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1702.4     | 1261.012 | 1.261  | 1654.0     | 1698.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.8       | 15.407   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1566.6     | 1160.469 | 1.160  | 1443.0     | 1543.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1703.7     | 1262.000 | 1.262  | 1573.0     | 1676.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 560.4      | 415.111  | 0.415  | 405.0      | 563.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.5      | 277.432  | 0.277  | 371.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1664.4     | 1232.926 | 1.233  | 1626.5                | 1664.4                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.6       | 17.519   | 0.018  | 20.8                  | 23.6                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1570.5     | 1163.358 | 1.163  | 1566.6                | 1570.5                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1704.1     | 1262.272 | 1.262  | 1703.7                | 1704.1                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 564.3      | 418.000  | 0.418  | 560.4                 | 564.3                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.5      | 277.370  | 0.277  | 374.4                 | 374.5                    |

Final op time: 1.262 us (1704.1 cycles), bucket=TRISC_0 compute

