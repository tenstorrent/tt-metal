# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_23_00_38
CCLs: broadcast

## broadcast

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_broadcast.py::test_ccl_broadcast
Configured benchmark shape: [1, 7168]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = slowest device's slowest top-level bucket, using per-device averages across iterations only.

Cross-device bucket averages are omitted for this CCL.

### num_links=2, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_03_23_00_47/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 79.5       | 58.864   | 0.059  | 71.0       | 81.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12428.4    | 9206.198 | 9.206  | 12248.0    | 12420.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12409.8    | 9192.420 | 9.192  | 11128.0    | 12462.0       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12310.7    | 9119.062 | 9.119  | 9565.0     | 12387.5       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12296.8    | 9108.716 | 9.109  | 7696.0     | 12459.5       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12370.6    | 9163.383 | 9.163  | 11709.0    | 12377.5       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12431.0    | 9208.123 | 9.208  | 11581.0    | 12466.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12334.9    | 9136.963 | 9.137  | 9832.0     | 12439.0       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 12333.2    | 9135.728 | 9.136  | 8377.0     | 12486.5       |

Final op time: 9.208 us (12431.0 cycles), device=5, bucket=NCRISC transport

