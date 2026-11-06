# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_27_52
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

Report: generated/profiler/reports/2026_11_04_09_28_05/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1650.3     | 1222.469 | 1.222  | 1593.0     | 1626.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 22.6       | 16.741   | 0.017  | 18.0       | 18.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1619.7     | 1199.778 | 1.200  | 1260.0     | 1611.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1751.5     | 1297.383 | 1.297  | 1396.0     | 1742.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 570.2      | 422.395  | 0.422  | 428.0      | 575.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 370.0      | 274.074  | 0.274  | 365.0      | 370.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1754.3     | 1299.457 | 1.299  | 1685.0     | 1730.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.8       | 13.901   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1610.2     | 1192.765 | 1.193  | 1427.0     | 1589.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1743.6     | 1291.556 | 1.292  | 1561.0     | 1721.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 574.3      | 425.432  | 0.425  | 423.0      | 575.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.2      | 273.506  | 0.274  | 365.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1702.3     | 1260.963 | 1.261  | 1650.3                | 1702.3                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 20.7       | 15.321   | 0.015  | 18.8                  | 20.7                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1615.0     | 1196.272 | 1.196  | 1610.2                | 1615.0                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1747.5     | 1294.469 | 1.294  | 1743.6                | 1747.5                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 572.3      | 423.914  | 0.424  | 570.2                 | 572.3                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.6      | 273.790  | 0.274  | 369.2                 | 369.6                    |

Final op time: 1.294 us (1747.5 cycles), bucket=TRISC_0 compute

