# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_36_45
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_02_09_36_57/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1618.2     | 1198.667 | 1.199  | 1575.0     | 1605.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.7       | 19.802   | 0.020  | 18.0       | 24.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1590.1     | 1177.827 | 1.178  | 1455.0     | 1585.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1686.2     | 1249.012 | 1.249  | 1551.0     | 1680.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 575.0      | 425.926  | 0.426  | 438.0      | 584.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.7      | 278.296  | 0.278  | 371.0      | 375.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1693.2     | 1254.198 | 1.254  | 1656.0     | 1689.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 21.0       | 15.531   | 0.016  | 18.0       | 20.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1581.5     | 1171.457 | 1.171  | 1460.0     | 1555.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1676.2     | 1241.630 | 1.242  | 1557.0     | 1644.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 566.3      | 419.457  | 0.419  | 442.0      | 573.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.7      | 277.580  | 0.278  | 370.0      | 375.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1655.7     | 1226.432 | 1.226  | 1618.2                | 1655.7                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.9       | 17.667   | 0.018  | 21.0                  | 23.9                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1585.8     | 1174.642 | 1.175  | 1581.5                | 1585.8                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1681.2     | 1245.321 | 1.245  | 1676.2                | 1681.2                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 570.6      | 422.691  | 0.423  | 566.3                 | 570.6                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.2      | 277.938  | 0.278  | 374.7                 | 375.2                    |

Final op time: 1.245 us (1681.2 cycles), bucket=TRISC_0 compute

