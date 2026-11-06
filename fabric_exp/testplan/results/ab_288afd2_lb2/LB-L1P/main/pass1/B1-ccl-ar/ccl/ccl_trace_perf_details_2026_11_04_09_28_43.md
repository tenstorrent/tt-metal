# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_28_43
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

Report: generated/profiler/reports/2026_11_04_09_28_54/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1642.8     | 1216.914 | 1.217  | 1586.0     | 1630.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 27.1       | 20.074   | 0.020  | 18.0       | 23.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1636.6     | 1212.321 | 1.212  | 1540.0     | 1617.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1728.5     | 1280.395 | 1.280  | 1620.0     | 1716.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.9      | 423.605  | 0.424  | 410.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 370.0      | 274.074  | 0.274  | 359.0      | 371.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1727.2     | 1279.383 | 1.279  | 1669.0     | 1696.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.5       | 13.728   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1622.9     | 1202.148 | 1.202  | 1370.0     | 1596.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1714.1     | 1269.704 | 1.270  | 1463.0     | 1684.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 574.5      | 425.580  | 0.426  | 440.0      | 575.0         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.6      | 273.753  | 0.274  | 364.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1685.0     | 1248.148 | 1.248  | 1642.8                | 1685.0                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.8       | 16.901   | 0.017  | 18.5                  | 22.8                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1629.8     | 1207.235 | 1.207  | 1622.9                | 1629.8                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1721.3     | 1275.049 | 1.275  | 1714.1                | 1721.3                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 573.2      | 424.593  | 0.425  | 571.9                 | 573.2                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.8      | 273.914  | 0.274  | 369.6                 | 369.8                    |

Final op time: 1.275 us (1721.3 cycles), bucket=TRISC_0 compute

