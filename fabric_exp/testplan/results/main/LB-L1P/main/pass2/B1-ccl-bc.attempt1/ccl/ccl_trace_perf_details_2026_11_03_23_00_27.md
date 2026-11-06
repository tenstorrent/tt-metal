# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_23_00_27
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

Report: generated/profiler/reports/2026_11_03_23_00_37/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 77.3       | 57.284   | 0.057  | 70.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5130.9     | 3800.667 | 3.801  | 5055.0     | 5104.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5250.4     | 3889.210 | 3.889  | 5052.0     | 5189.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5168.5     | 3828.543 | 3.829  | 4546.0     | 5184.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5190.7     | 3844.988 | 3.845  | 4244.0     | 5177.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5180.3     | 3837.284 | 3.837  | 4981.0     | 5123.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5290.9     | 3919.185 | 3.919  | 5076.0     | 5191.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5216.6     | 3864.173 | 3.864  | 5009.0     | 5166.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5241.5     | 3882.593 | 3.883  | 5047.0     | 5249.0        |

Final op time: 3.919 us (5290.9 cycles), device=5, bucket=NCRISC transport

