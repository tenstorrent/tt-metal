# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_28_55
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

Report: generated/profiler/reports/2026_11_04_05_29_05/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE              | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_SENDER_WRITER | 30    | 1650.0     | 1222.247 | 1.222  | 1600.0     | 1632.5        |
| 0         | NCRISC              | CCL_SENDER_WRITER | 30    | 26.6       | 19.704   | 0.020  | 18.0       | 24.5          |
| 0         | NCRISC              | CCL_RECEIVER      | 30    | 1685.2     | 1248.296 | 1.248  | 1499.0     | 1657.5        |
| 0         | TRISC_0             | CCL_COMPUTE       | 30    | 1777.1     | 1316.370 | 1.316  | 1594.0     | 1748.5        |
| 0         | TRISC_1             | CCL_COMPUTE       | 30    | 571.8      | 423.580  | 0.424  | 447.0      | 575.0         |
| 0         | TRISC_2             | CCL_COMPUTE       | 30    | 369.1      | 273.432  | 0.273  | 364.0      | 369.0         |
| 1         | BRISC               | CCL_SENDER_WRITER | 30    | 1782.3     | 1320.198 | 1.320  | 1689.0     | 1773.5        |
| 1         | NCRISC              | CCL_SENDER_WRITER | 30    | 19.1       | 14.123   | 0.014  | 18.0       | 18.0          |
| 1         | NCRISC              | CCL_RECEIVER      | 30    | 1660.5     | 1230.025 | 1.230  | 1472.0     | 1671.0        |
| 1         | TRISC_0             | CCL_COMPUTE       | 30    | 1755.1     | 1300.074 | 1.300  | 1567.0     | 1765.5        |
| 1         | TRISC_1             | CCL_COMPUTE       | 30    | 569.9      | 422.173  | 0.422  | 439.0      | 570.5         |
| 1         | TRISC_2             | CCL_COMPUTE       | 30    | 369.6      | 273.802  | 0.274  | 366.0      | 369.0         |

#### Bucket averages across devices

| RISC processor type | ZONE              | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ----------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| BRISC               | CCL_SENDER_WRITER | 2            | 1716.2     | 1271.222 | 1.271  | 1650.0                | 1716.2                   |
| NCRISC              | CCL_SENDER_WRITER | 2            | 22.8       | 16.914   | 0.017  | 19.1                  | 22.8                     |
| NCRISC              | CCL_RECEIVER      | 2            | 1672.9     | 1239.160 | 1.239  | 1660.5                | 1672.9                   |
| TRISC_0             | CCL_COMPUTE       | 2            | 1766.1     | 1308.222 | 1.308  | 1755.1                | 1766.1                   |
| TRISC_1             | CCL_COMPUTE       | 2            | 570.9      | 422.877  | 0.423  | 569.9                 | 570.9                    |
| TRISC_2             | CCL_COMPUTE       | 2            | 369.4      | 273.617  | 0.274  | 369.1                 | 369.4                    |

Final op time: 1.308 us (1766.1 cycles), bucket=TRISC_0 compute

