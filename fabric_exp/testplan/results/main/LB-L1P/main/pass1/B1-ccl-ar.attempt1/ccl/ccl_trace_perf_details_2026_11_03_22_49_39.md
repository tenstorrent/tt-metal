# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_49_39
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_03_22_49_55/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1648.1     | 1220.840 | 1.221  | 1587.0     | 1659.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.4       | 19.531   | 0.020  | 18.0       | 22.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1600.1     | 1185.284 | 1.185  | 1518.0     | 1575.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1732.4     | 1283.259 | 1.283  | 1654.0     | 1705.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.4      | 423.259  | 0.423  | 416.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 368.4      | 272.864  | 0.273  | 362.0      | 368.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1722.7     | 1276.049 | 1.276  | 1669.0     | 1691.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.6       | 13.753   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1581.1     | 1171.210 | 1.171  | 1407.0     | 1559.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1720.8     | 1274.642 | 1.275  | 1537.0     | 1704.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 568.0      | 420.741  | 0.421  | 428.0      | 573.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.9      | 273.235  | 0.273  | 363.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1685.4     | 1248.444 | 1.248  | 1648.1                | 1685.4                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.5       | 16.642   | 0.017  | 18.6                  | 22.5                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1590.6     | 1178.247 | 1.178  | 1581.1                | 1590.6                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1726.6     | 1278.951 | 1.279  | 1720.8                | 1726.6                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 569.7      | 422.000  | 0.422  | 568.0                 | 569.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 368.6      | 273.049  | 0.273  | 368.4                 | 368.6                    |

Final op time: 1.279 us (1726.6 cycles), bucket=TRISC_0 compute

