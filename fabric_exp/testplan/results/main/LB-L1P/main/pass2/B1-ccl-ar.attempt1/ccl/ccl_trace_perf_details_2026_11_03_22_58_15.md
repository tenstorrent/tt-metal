# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_58_15
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

Report: generated/profiler/reports/2026_11_03_22_58_28/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1768.0     | 1309.630 | 1.310  | 1718.0     | 1764.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 1750.4     | 1296.568 | 1.297  | 1642.0     | 1756.5        |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1730.7     | 1282.025 | 1.282  | 1403.0     | 1727.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1822.2     | 1349.778 | 1.350  | 1480.0     | 1814.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.5      | 423.358  | 0.423  | 417.0      | 574.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.6      | 273.778  | 0.274  | 363.0      | 369.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1846.9     | 1368.099 | 1.368  | 1776.0     | 1832.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 1766.8     | 1308.716 | 1.309  | 1670.0     | 1771.0        |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1726.5     | 1278.914 | 1.279  | 1530.0     | 1703.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1821.9     | 1349.531 | 1.350  | 1623.0     | 1792.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 568.1      | 420.790  | 0.421  | 445.0      | 573.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.7      | 273.111  | 0.273  | 358.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1807.5     | 1338.864 | 1.339  | 1768.0                | 1807.5                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 1758.6     | 1302.642 | 1.303  | 1750.4                | 1758.6                   |
| NCRISC              | CCL_RECEIVER      | 2            | 1728.6     | 1280.469 | 1.280  | 1726.5                | 1728.6                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1822.0     | 1349.654 | 1.350  | 1821.9                | 1822.0                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 569.8      | 422.074  | 0.422  | 568.1                 | 569.8                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.1      | 273.444  | 0.273  | 368.7                 | 369.1                    |

Final op time: 1.350 us (1822.0 cycles), bucket=TRISC_0 compute

