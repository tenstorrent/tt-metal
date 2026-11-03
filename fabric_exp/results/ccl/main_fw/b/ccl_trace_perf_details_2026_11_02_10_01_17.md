# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_01_17
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

Report: generated/profiler/reports/2026_11_02_10_01_26/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1636.6     | 1212.296 | 1.212  | 1575.0     | 1640.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 29.6       | 21.926   | 0.022  | 18.0       | 25.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1633.9     | 1210.296 | 1.210  | 1462.0     | 1624.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1727.0     | 1279.235 | 1.279  | 1565.0     | 1718.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 581.4      | 430.642  | 0.431  | 416.0      | 592.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 375.4      | 278.074  | 0.278  | 371.0      | 374.5         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1732.8     | 1283.556 | 1.284  | 1654.0     | 1728.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 21.2       | 15.679   | 0.016  | 18.0       | 20.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1620.8     | 1200.617 | 1.201  | 1373.0     | 1621.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1716.8     | 1271.679 | 1.272  | 1478.0     | 1714.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 567.4      | 420.272  | 0.420  | 441.0      | 575.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 375.5      | 278.123  | 0.278  | 372.0      | 375.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1684.7     | 1247.926 | 1.248  | 1636.6                | 1684.7                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 25.4       | 18.802   | 0.019  | 21.2                  | 25.4                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1627.4     | 1205.457 | 1.205  | 1620.8                | 1627.4                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1721.9     | 1275.457 | 1.275  | 1716.8                | 1721.9                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 574.4      | 425.457  | 0.425  | 567.4                 | 574.4                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 375.4      | 278.099  | 0.278  | 375.4                 | 375.4                    |

Final op time: 1.275 us (1721.9 cycles), bucket=TRISC_0 compute

