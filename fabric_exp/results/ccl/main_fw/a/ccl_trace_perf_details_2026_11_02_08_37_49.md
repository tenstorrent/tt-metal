# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_08_37_49
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

Report: generated/profiler/reports/2026_11_02_08_38_05/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | ALLGATHER_GATHER    | 30    | 3577.9     | 2650.321 | 2.650  | 2104.0     | 3514.0        |
| 0         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1882.8     | 1394.642 | 1.395  | 1765.0     | 1846.0        |
| 0         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3511.8     | 2601.358 | 2.601  | 2139.0     | 3632.0        |
| 1         | NCRISC              | ALLGATHER_GATHER    | 30    | 3616.5     | 2678.914 | 2.679  | 2046.0     | 3800.0        |
| 1         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3534.1     | 2617.877 | 2.618  | 2016.0     | 3475.5        |
| 1         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1887.3     | 1398.000 | 1.398  | 1798.0     | 1871.5        |
| 2         | NCRISC              | ALLGATHER_GATHER    | 30    | 3590.4     | 2659.580 | 2.660  | 2062.0     | 4119.0        |
| 2         | BRISC               | ALLGATHER_TRANSPORT | 30    | 3464.1     | 2566.025 | 2.566  | 2095.0     | 3436.0        |
| 2         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 1985.4     | 1470.642 | 1.471  | 1795.0     | 1960.0        |
| 3         | NCRISC              | ALLGATHER_GATHER    | 30    | 3547.0     | 2627.432 | 2.627  | 1834.0     | 3485.0        |
| 3         | BRISC               | ALLGATHER_TRANSPORT | 30    | 1922.9     | 1424.370 | 1.424  | 1840.0     | 1920.0        |
| 3         | NCRISC              | ALLGATHER_TRANSPORT | 30    | 3502.3     | 2594.272 | 2.594  | 2060.0     | 3505.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | ------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | ALLGATHER_GATHER    | 4            | 3583.0     | 2654.062 | 2.654  | 3547.0                | 3584.2                   |
| BRISC               | ALLGATHER_TRANSPORT | 4            | 2701.0     | 2000.728 | 2.001  | 1882.8                | 2693.5                   |
| NCRISC              | ALLGATHER_TRANSPORT | 4            | 2721.7     | 2016.068 | 2.016  | 1887.3                | 2743.8                   |

Final op time: 2.654 us (3583.0 cycles), bucket=NCRISC gather

