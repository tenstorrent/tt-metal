# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_34_31
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

Report: generated/profiler/reports/2026_11_04_09_34_49/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1739.8     | 1288.765 | 1.289  | 1699.0     | 1729.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.5       | 13.679   | 0.014  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1585.7     | 1174.617 | 1.175  | 1154.0     | 1591.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1723.0     | 1276.272 | 1.276  | 1277.0     | 1732.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 572.3      | 423.901  | 0.424  | 471.0      | 572.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.2      | 273.506  | 0.274  | 364.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1680.7     | 1244.938 | 1.245  | 1620.0     | 1683.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 36.9       | 27.358   | 0.027  | 21.0       | 36.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1643.8     | 1217.654 | 1.218  | 1548.0     | 1597.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1782.2     | 1320.123 | 1.320  | 1687.0     | 1741.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 573.0      | 424.469  | 0.424  | 428.0      | 577.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 370.1      | 274.173  | 0.274  | 363.0      | 370.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1710.2     | 1266.852 | 1.267  | 1680.7                | 1710.2                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 27.7       | 20.519   | 0.021  | 18.5                  | 27.7                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1614.8     | 1196.136 | 1.196  | 1585.7                | 1614.8                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1752.6     | 1298.198 | 1.298  | 1723.0                | 1752.6                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.6      | 424.185  | 0.424  | 572.3                 | 572.6                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.7      | 273.840  | 0.274  | 369.2                 | 369.7                    |

Final op time: 1.298 us (1752.6 cycles), bucket=TRISC_0 compute

