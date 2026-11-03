# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_21_34
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_02_09_21_47/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1754.7     | 1299.802 | 1.300  | 1703.0     | 1751.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 1732.3     | 1283.210 | 1.283  | 1667.0     | 1725.0        |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1718.5     | 1272.938 | 1.273  | 1507.0     | 1719.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1810.5     | 1341.111 | 1.341  | 1613.0     | 1810.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 572.7      | 424.247  | 0.424  | 406.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.1      | 277.852  | 0.278  | 368.0      | 374.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1839.8     | 1362.815 | 1.363  | 1794.0     | 1839.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 1724.3     | 1277.284 | 1.277  | 1681.0     | 1721.5        |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1713.4     | 1269.160 | 1.269  | 1611.0     | 1679.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1800.0     | 1333.309 | 1.333  | 1666.0     | 1768.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 569.2      | 421.630  | 0.422  | 449.0      | 574.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.1      | 277.852  | 0.278  | 371.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1797.3     | 1331.309 | 1.331  | 1754.7                | 1797.3                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 1728.3     | 1280.247 | 1.280  | 1724.3                | 1728.3                   |
| NCRISC              | CCL_RECEIVER      | 2            | 1715.9     | 1271.049 | 1.271  | 1713.4                | 1715.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1805.2     | 1337.210 | 1.337  | 1800.0                | 1805.2                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.0      | 422.938  | 0.423  | 569.2                 | 571.0                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.1      | 277.852  | 0.278  | 375.1                 | 375.1                    |

Final op time: 1.337 us (1805.2 cycles), bucket=TRISC_0 compute

