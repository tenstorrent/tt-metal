# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_35_24
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

Report: generated/profiler/reports/2026_11_04_05_35_34/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 78.1       | 57.827   | 0.058  | 71.0       | 79.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5130.8     | 3800.568 | 3.801  | 5051.0     | 5112.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5268.0     | 3902.222 | 3.902  | 5004.0     | 5197.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5131.2     | 3800.864 | 3.801  | 3657.0     | 5138.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5120.0     | 3792.593 | 3.793  | 2612.0     | 5147.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5156.5     | 3819.654 | 3.820  | 4954.0     | 5136.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5291.8     | 3919.877 | 3.920  | 5028.0     | 5190.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5166.3     | 3826.889 | 3.827  | 4627.0     | 5139.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5162.1     | 3823.802 | 3.824  | 3817.0     | 5164.0        |

Final op time: 3.920 us (5291.8 cycles), device=5, bucket=NCRISC transport

