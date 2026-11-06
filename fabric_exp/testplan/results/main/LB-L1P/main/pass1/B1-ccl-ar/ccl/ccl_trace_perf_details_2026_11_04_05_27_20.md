# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_27_20
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

Report: generated/profiler/reports/2026_11_04_05_27_36/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1642.3     | 1216.543 | 1.217  | 1593.0     | 1641.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 28.7       | 21.235   | 0.021  | 18.0       | 24.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1604.6     | 1188.568 | 1.189  | 1518.0     | 1565.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1738.6     | 1287.877 | 1.288  | 1644.0     | 1709.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.7      | 423.481  | 0.423  | 434.0      | 574.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 370.0      | 274.074  | 0.274  | 366.0      | 370.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1734.7     | 1284.988 | 1.285  | 1670.0     | 1726.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.7       | 14.593   | 0.015  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1587.8     | 1176.173 | 1.176  | 1393.0     | 1555.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1722.8     | 1276.173 | 1.276  | 1524.0     | 1684.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 572.1      | 423.778  | 0.424  | 455.0      | 574.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.9      | 273.975  | 0.274  | 360.0      | 370.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1688.5     | 1250.765 | 1.251  | 1642.3                | 1688.5                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 24.2       | 17.914   | 0.018  | 19.7                  | 24.2                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1596.2     | 1182.370 | 1.182  | 1587.8                | 1596.2                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1730.7     | 1282.025 | 1.282  | 1722.8                | 1730.7                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.9      | 423.630  | 0.424  | 571.7                 | 571.9                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.9      | 274.025  | 0.274  | 369.9                 | 369.9                    |

Final op time: 1.282 us (1730.7 cycles), bucket=TRISC_0 compute

