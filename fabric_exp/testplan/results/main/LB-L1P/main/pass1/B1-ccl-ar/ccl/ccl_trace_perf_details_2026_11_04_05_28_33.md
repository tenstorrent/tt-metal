# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_28_33
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

Report: generated/profiler/reports/2026_11_04_05_28_44/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1641.7     | 1216.049 | 1.216  | 1585.0     | 1632.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 23.2       | 17.210   | 0.017  | 18.0       | 20.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1629.2     | 1206.790 | 1.207  | 1239.0     | 1607.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1722.8     | 1276.123 | 1.276  | 1324.0     | 1705.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 567.3      | 420.198  | 0.420  | 405.0      | 574.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.9      | 273.975  | 0.274  | 365.0      | 369.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1727.0     | 1279.259 | 1.279  | 1648.0     | 1691.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.8       | 14.667   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1622.4     | 1201.753 | 1.202  | 1428.0     | 1576.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1714.2     | 1269.778 | 1.270  | 1530.0     | 1670.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.0      | 423.728  | 0.424  | 468.0      | 575.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.0      | 273.333  | 0.273  | 364.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1684.3     | 1247.654 | 1.248  | 1641.7                | 1684.3                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 21.5       | 15.938   | 0.016  | 19.8                  | 21.5                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1625.8     | 1204.272 | 1.204  | 1622.4                | 1625.8                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1718.5     | 1272.951 | 1.273  | 1714.2                | 1718.5                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 569.6      | 421.963  | 0.422  | 567.3                 | 569.6                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.4      | 273.654  | 0.274  | 369.0                 | 369.4                    |

Final op time: 1.273 us (1718.5 cycles), bucket=TRISC_0 compute

