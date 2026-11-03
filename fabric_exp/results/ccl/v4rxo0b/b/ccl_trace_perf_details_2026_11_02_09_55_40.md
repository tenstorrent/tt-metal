# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_55_40
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

### num_links=1, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_02_09_55_53/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.4       | 55.111   | 0.055  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6733.1     | 4987.506 | 4.988  | 6639.0     | 6735.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6824.8     | 5055.432 | 5.055  | 6684.0     | 6809.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6734.0     | 4988.173 | 4.988  | 5439.0     | 6786.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6737.1     | 4990.469 | 4.990  | 4451.0     | 6821.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6728.4     | 4983.975 | 4.984  | 6554.0     | 6738.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6840.6     | 5067.111 | 5.067  | 6694.0     | 6817.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6736.4     | 4989.926 | 4.990  | 5699.0     | 6761.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6743.4     | 4995.111 | 4.995  | 4744.0     | 6803.5        |

Final op time: 5.067 us (6840.6 cycles), device=5, bucket=NCRISC transport

