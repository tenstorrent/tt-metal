# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_35_03
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

Report: generated/profiler/reports/2026_11_04_09_35_17/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1742.7     | 1290.864 | 1.291  | 1721.0     | 1738.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.4       | 13.630   | 0.014  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1626.9     | 1205.136 | 1.205  | 1279.0     | 1637.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1721.4     | 1275.111 | 1.275  | 1383.0     | 1735.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 569.5      | 421.852  | 0.422  | 464.0      | 571.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 370.0      | 274.049  | 0.274  | 367.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1676.3     | 1241.704 | 1.242  | 1602.0     | 1681.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 31.6       | 23.432   | 0.023  | 18.0       | 31.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1684.3     | 1247.605 | 1.248  | 1548.0     | 1644.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1777.9     | 1316.938 | 1.317  | 1643.0     | 1739.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.3      | 423.926  | 0.424  | 421.0      | 575.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.2      | 273.506  | 0.274  | 365.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1709.5     | 1266.284 | 1.266  | 1676.3                | 1709.5                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 25.0       | 18.531   | 0.019  | 18.4                  | 25.0                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1655.6     | 1226.370 | 1.226  | 1626.9                | 1655.6                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1749.6     | 1296.025 | 1.296  | 1721.4                | 1749.6                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 570.9      | 422.889  | 0.423  | 569.5                 | 570.9                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.6      | 273.778  | 0.274  | 369.2                 | 369.6                    |

Final op time: 1.296 us (1749.6 cycles), bucket=TRISC_0 compute

