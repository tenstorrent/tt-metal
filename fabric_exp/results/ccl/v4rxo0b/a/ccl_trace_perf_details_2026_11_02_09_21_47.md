# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_21_47
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

Report: generated/profiler/reports/2026_11_02_09_22_00/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1622.9     | 1202.123 | 1.202  | 1575.0     | 1612.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 24.2       | 17.926   | 0.018  | 18.0       | 20.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1591.3     | 1178.716 | 1.179  | 1344.0     | 1587.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1686.6     | 1249.309 | 1.249  | 1444.0     | 1684.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 576.6      | 427.111  | 0.427  | 447.0      | 579.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 374.5      | 277.383  | 0.277  | 368.0      | 374.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1695.0     | 1255.531 | 1.256  | 1657.0     | 1691.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 21.9       | 16.198   | 0.016  | 18.0       | 20.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1580.9     | 1171.012 | 1.171  | 1449.0     | 1556.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1675.3     | 1240.938 | 1.241  | 1541.0     | 1649.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 571.7      | 423.457  | 0.423  | 464.0      | 580.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 374.5      | 277.432  | 0.277  | 369.0      | 374.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1658.9     | 1228.827 | 1.229  | 1622.9                | 1658.9                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 23.0       | 17.062   | 0.017  | 21.9                  | 23.0                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1586.1     | 1174.864 | 1.175  | 1580.9                | 1586.1                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1680.9     | 1245.123 | 1.245  | 1675.3                | 1680.9                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 574.1      | 425.284  | 0.425  | 571.7                 | 574.1                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 374.5      | 277.407  | 0.277  | 374.5                 | 374.5                    |

Final op time: 1.245 us (1680.9 cycles), bucket=TRISC_0 compute

