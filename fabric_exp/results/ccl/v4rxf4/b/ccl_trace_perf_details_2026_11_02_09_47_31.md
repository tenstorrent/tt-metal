# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_47_31
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

Report: generated/profiler/reports/2026_11_02_09_47_44/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1753.0     | 1298.543 | 1.299  | 1692.0     | 1750.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 1737.4     | 1286.938 | 1.287  | 1666.0     | 1731.5        |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1708.5     | 1265.556 | 1.266  | 1340.0     | 1724.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1800.1     | 1333.432 | 1.333  | 1452.0     | 1821.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.6      | 423.383  | 0.423  | 420.0      | 574.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.2      | 277.951  | 0.278  | 368.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1832.3     | 1357.259 | 1.357  | 1787.0     | 1827.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 1726.7     | 1279.062 | 1.279  | 1671.0     | 1720.5        |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1713.9     | 1269.531 | 1.270  | 1550.0     | 1691.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1807.1     | 1338.568 | 1.339  | 1635.0     | 1782.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 565.9      | 419.210  | 0.419  | 403.0      | 569.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.6      | 277.506  | 0.278  | 367.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1792.7     | 1327.901 | 1.328  | 1753.0                | 1792.7                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 1732.0     | 1283.000 | 1.283  | 1726.7                | 1732.0                   |
| NCRISC              | CCL_RECEIVER      | 2            | 1711.2     | 1267.543 | 1.268  | 1708.5                | 1711.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1803.6     | 1336.000 | 1.336  | 1800.1                | 1803.6                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 568.8      | 421.296  | 0.421  | 565.9                 | 568.8                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.9      | 277.728  | 0.278  | 374.6                 | 374.9                    |

Final op time: 1.336 us (1803.6 cycles), bucket=TRISC_0 compute

