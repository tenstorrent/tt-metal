# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_09_28_31
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

Report: generated/profiler/reports/2026_11_04_09_28_45/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1764.1     | 1306.716 | 1.307  | 1678.0     | 1791.0        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 18.9       | 13.975   | 0.014  | 18.0       | 18.0          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1597.3     | 1183.160 | 1.183  | 1175.0     | 1621.0        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1732.4     | 1283.259 | 1.283  | 1308.0     | 1751.0        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 573.9      | 425.111  | 0.425  | 434.0      | 577.5         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.3      | 273.580  | 0.274  | 366.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1679.0     | 1243.704 | 1.244  | 1639.0     | 1676.0        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 31.8       | 23.580   | 0.024  | 18.0       | 31.5          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1672.5     | 1238.889 | 1.239  | 1529.0     | 1651.5        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1810.1     | 1340.815 | 1.341  | 1660.0     | 1789.0        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 569.7      | 422.000  | 0.422  | 473.0      | 571.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.1      | 273.432  | 0.273  | 363.0      | 368.5         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1721.5     | 1275.210 | 1.275  | 1679.0                | 1721.5                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 25.4       | 18.778   | 0.019  | 18.9                  | 25.4                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1634.9     | 1211.025 | 1.211  | 1597.3                | 1634.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1771.2     | 1312.037 | 1.312  | 1732.4                | 1771.2                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 571.8      | 423.556  | 0.424  | 569.7                 | 571.8                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.2      | 273.506  | 0.274  | 369.1                 | 369.2                    |

Final op time: 1.312 us (1771.2 cycles), bucket=TRISC_0 compute

