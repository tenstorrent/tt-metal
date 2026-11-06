# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_28_33
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

Report: generated/profiler/reports/2026_11_04_09_28_43/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1766.9     | 1308.790 | 1.309  | 1723.0     | 1763.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 1751.9     | 1297.704 | 1.298  | 1653.0     | 1748.0        |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1723.6     | 1276.741 | 1.277  | 1340.0     | 1737.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1816.6     | 1345.630 | 1.346  | 1436.0     | 1831.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 573.1      | 424.519  | 0.425  | 451.0      | 576.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.0      | 273.309  | 0.273  | 359.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1847.8     | 1368.716 | 1.369  | 1802.0     | 1833.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 1756.8     | 1301.358 | 1.301  | 1664.0     | 1763.0        |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1727.6     | 1279.728 | 1.280  | 1537.0     | 1712.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1823.0     | 1350.395 | 1.350  | 1657.0     | 1805.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 570.1      | 422.321  | 0.422  | 420.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.6      | 273.012  | 0.273  | 359.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1807.3     | 1338.753 | 1.339  | 1766.9                | 1807.3                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 1754.4     | 1299.531 | 1.300  | 1751.9                | 1754.4                   |
| NCRISC              | CCL_RECEIVER      | 2            | 1725.6     | 1278.235 | 1.278  | 1723.6                | 1725.6                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1819.8     | 1348.012 | 1.348  | 1816.6                | 1819.8                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.6      | 423.420  | 0.423  | 570.1                 | 571.6                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 368.8      | 273.160  | 0.273  | 368.6                 | 368.8                    |

Final op time: 1.348 us (1819.8 cycles), bucket=TRISC_0 compute

