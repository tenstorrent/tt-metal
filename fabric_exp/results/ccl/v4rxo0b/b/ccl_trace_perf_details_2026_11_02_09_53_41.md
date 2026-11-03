# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_53_41
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

Report: generated/profiler/reports/2026_11_02_09_53_55/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1640.8     | 1215.407 | 1.215  | 1574.0     | 1621.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 29.7       | 22.025   | 0.022  | 18.0       | 26.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1582.4     | 1172.123 | 1.172  | 1480.0     | 1585.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1715.2     | 1270.494 | 1.270  | 1638.0     | 1714.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 564.6      | 418.198  | 0.418  | 402.0      | 570.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.0      | 277.778  | 0.278  | 365.0      | 375.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1701.5     | 1260.370 | 1.260  | 1662.0     | 1688.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.9       | 14.716   | 0.015  | 18.0       | 19.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1558.6     | 1154.519 | 1.155  | 1151.0     | 1547.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1690.5     | 1252.198 | 1.252  | 1271.0     | 1672.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 560.8      | 415.432  | 0.415  | 392.0      | 565.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.0      | 277.012  | 0.277  | 369.0      | 374.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1671.2     | 1237.889 | 1.238  | 1640.8                | 1671.2                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.8       | 18.370   | 0.018  | 19.9                  | 24.8                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1570.5     | 1163.321 | 1.163  | 1558.6                | 1570.5                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1702.8     | 1261.346 | 1.261  | 1690.5                | 1702.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 562.7      | 416.815  | 0.417  | 560.8                 | 562.7                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.5      | 277.395  | 0.277  | 374.0                 | 374.5                    |

Final op time: 1.261 us (1702.8 cycles), bucket=TRISC_0 compute

