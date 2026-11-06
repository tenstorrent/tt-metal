# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_37_40
CCLs: all_gather

## all_gather

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_gather.py::test_ccl_all_gather
Configured benchmark shape: [1, 7168]
Configured transport mode: single-link only
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_links=1, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_04_05_37_50/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3555.6     | 2633.778 | 2.634  | 2388.0     | 3563.5        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1880.8     | 1393.185 | 1.393  | 1789.0     | 1837.5        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3481.9     | 2579.210 | 2.579  | 2389.0     | 3443.5        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3574.5     | 2647.778 | 2.648  | 2449.0     | 3486.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3496.6     | 2590.049 | 2.590  | 2286.0     | 3470.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1864.5     | 1381.086 | 1.381  | 1782.0     | 1869.0        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3537.2     | 2620.123 | 2.620  | 2363.0     | 3410.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3442.3     | 2549.877 | 2.550  | 2463.0     | 3394.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1955.3     | 1448.370 | 1.448  | 1815.0     | 1922.5        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3516.7     | 2604.938 | 2.605  | 2522.0     | 3461.5        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1920.2     | 1422.346 | 1.422  | 1818.0     | 1910.5        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3471.7     | 2571.654 | 2.572  | 1992.0     | 3376.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3546.0     | 2626.654 | 2.627  | 3516.7                | 3546.4                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2685.0     | 1988.864 | 1.989  | 1880.8                | 2681.2                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2693.4     | 1995.080 | 1.995  | 1864.5                | 2713.5                   |

Final op time: 2.627 us (3546.0 cycles), bucket=NCRISC gather

