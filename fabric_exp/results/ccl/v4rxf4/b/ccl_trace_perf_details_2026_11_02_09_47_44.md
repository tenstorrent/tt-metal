# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_47_44
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

Report: generated/profiler/reports/2026_11_02_09_47_57/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1630.0     | 1207.407 | 1.207  | 1572.0     | 1621.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.1       | 19.309   | 0.019  | 18.0       | 23.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1602.5     | 1187.012 | 1.187  | 1436.0     | 1603.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1696.1     | 1256.370 | 1.256  | 1546.0     | 1697.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 577.1      | 427.457  | 0.427  | 388.0      | 589.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.5      | 278.148  | 0.278  | 366.0      | 375.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1700.9     | 1259.951 | 1.260  | 1655.0     | 1693.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 20.9       | 15.481   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1587.6     | 1176.000 | 1.176  | 1433.0     | 1570.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1684.7     | 1247.926 | 1.248  | 1517.0     | 1672.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 570.3      | 422.444  | 0.422  | 465.0      | 581.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.1      | 277.852  | 0.278  | 368.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1665.5     | 1233.679 | 1.234  | 1630.0                | 1665.5                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.5       | 17.395   | 0.017  | 20.9                  | 23.5                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1595.0     | 1181.506 | 1.182  | 1587.6                | 1595.0                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1690.4     | 1252.148 | 1.252  | 1684.7                | 1690.4                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 573.7      | 424.951  | 0.425  | 570.3                 | 573.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.3      | 278.000  | 0.278  | 375.1                 | 375.3                    |

Final op time: 1.252 us (1690.4 cycles), bucket=TRISC_0 compute

