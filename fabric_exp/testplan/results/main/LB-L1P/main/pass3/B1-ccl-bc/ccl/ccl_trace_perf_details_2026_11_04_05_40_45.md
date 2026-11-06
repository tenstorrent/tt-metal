# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_40_45
CCLs: broadcast

## broadcast

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_broadcast.py::test_ccl_broadcast
Configured benchmark shape: [1, 7168]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = slowest device's slowest top-level bucket, using per-device averages across iterations only.

Cross-device bucket averages are omitted for this CCL.

### num_links=1, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_04_05_40_55/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 79.0       | 58.494   | 0.058  | 72.0       | 79.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5130.5     | 3800.370 | 3.800  | 4986.0     | 5123.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5255.0     | 3892.568 | 3.893  | 5075.0     | 5195.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5149.1     | 3814.148 | 3.814  | 3825.0     | 5151.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5159.1     | 3821.580 | 3.822  | 3405.0     | 5251.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5152.7     | 3816.815 | 3.817  | 4845.0     | 5193.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5297.7     | 3924.198 | 3.924  | 4919.0     | 5223.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5178.3     | 3835.778 | 3.836  | 4760.0     | 5154.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5183.2     | 3839.407 | 3.839  | 4236.0     | 5170.0        |

Final op time: 3.924 us (5297.7 cycles), device=5, bucket=NCRISC transport

