# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_34_16
CCLs: all_reduce

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=2, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_04_05_34_25/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1644.2     | 1217.926 | 1.218  | 1593.0     | 1638.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.6       | 19.704   | 0.020  | 18.0       | 24.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1636.9     | 1212.519 | 1.213  | 1491.0     | 1633.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1729.0     | 1280.765 | 1.281  | 1584.0     | 1731.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.5      | 423.358  | 0.423  | 455.0      | 572.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.1      | 273.432  | 0.273  | 363.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1731.8     | 1282.840 | 1.283  | 1672.0     | 1730.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.0       | 14.099   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1620.0     | 1200.000 | 1.200  | 1441.0     | 1623.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1712.2     | 1268.272 | 1.268  | 1537.0     | 1716.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 569.4      | 421.778  | 0.422  | 442.0      | 572.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 368.9      | 273.259  | 0.273  | 364.0      | 368.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1688.0     | 1250.383 | 1.250  | 1644.2                | 1688.0                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.8       | 16.901   | 0.017  | 19.0                  | 22.8                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1628.5     | 1206.259 | 1.206  | 1620.0                | 1628.5                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1720.6     | 1274.519 | 1.275  | 1712.2                | 1720.6                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 570.5      | 422.568  | 0.423  | 569.4                 | 570.5                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.0      | 273.346  | 0.273  | 368.9                 | 369.0                    |

Final op time: 1.275 us (1720.6 cycles), bucket=TRISC_0 compute

