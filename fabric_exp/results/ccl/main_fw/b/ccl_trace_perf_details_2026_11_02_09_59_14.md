# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_59_14
CCLs: all_gather

## all_gather

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_gather.py::test_ccl_all_gather
Configured benchmark shape: [1, 7168]
Configured transport mode: single-link only
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_02_09_59_24/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3546.6     | 2627.111 | 2.627  | 2100.0     | 3533.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1861.9     | 1379.185 | 1.379  | 1762.0     | 1824.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3489.7     | 2584.988 | 2.585  | 2205.0     | 3435.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3589.5     | 2658.889 | 2.659  | 2246.0     | 3477.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3503.1     | 2594.864 | 2.595  | 2024.0     | 3498.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1877.8     | 1390.938 | 1.391  | 1778.0     | 1881.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3531.7     | 2616.049 | 2.616  | 2157.0     | 3627.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3422.3     | 2535.037 | 2.535  | 2027.0     | 3463.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1982.0     | 1468.123 | 1.468  | 1839.0     | 1959.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3516.2     | 2604.617 | 2.605  | 2146.0     | 3543.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1906.6     | 1412.296 | 1.412  | 1809.0     | 1902.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3462.8     | 2565.062 | 2.565  | 1990.0     | 3513.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3546.0     | 2626.667 | 2.627  | 3516.2                | 3539.1                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2673.5     | 1980.346 | 1.980  | 1861.9                | 2664.4                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2703.1     | 2002.278 | 2.002  | 1877.8                | 2722.4                   |

Final op time: 2.627 us (3546.0 cycles), bucket=NCRISC gather

