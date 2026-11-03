# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_24_04
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

### num_links=2, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_02_09_24_18/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.1       | 56.370   | 0.056  | 70.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10572.5    | 7831.457 | 7.831  | 10457.0    | 10552.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10619.7    | 7866.420 | 7.866  | 10389.0    | 10600.0       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10472.9    | 7757.704 | 7.758  | 7305.0     | 10563.5       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10442.8    | 7735.383 | 7.735  | 5078.0     | 10598.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10518.8    | 7791.704 | 7.792  | 9363.0     | 10529.5       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10622.5    | 7868.494 | 7.868  | 10322.0    | 10625.5       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10471.1    | 7756.395 | 7.756  | 7297.0     | 10533.0       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10453.2    | 7743.136 | 7.743  | 5345.0     | 10607.0       |

Final op time: 7.868 us (10622.5 cycles), device=5, bucket=NCRISC transport

