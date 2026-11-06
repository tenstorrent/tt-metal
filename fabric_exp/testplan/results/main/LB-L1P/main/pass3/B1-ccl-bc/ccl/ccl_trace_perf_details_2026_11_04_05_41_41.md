# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_41_41
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

### num_links=2, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_04_05_41_50/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 77.1       | 57.086   | 0.057  | 71.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5483.2     | 4061.630 | 4.062  | 5351.0     | 5480.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5571.6     | 4127.111 | 4.127  | 5332.0     | 5527.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5434.8     | 4025.753 | 4.026  | 4814.0     | 5439.0        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5486.3     | 4063.951 | 4.064  | 4526.0     | 5498.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5461.3     | 4045.432 | 4.045  | 5193.0     | 5428.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5594.6     | 4144.123 | 4.144  | 5282.0     | 5529.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5469.8     | 4051.679 | 4.052  | 5181.0     | 5460.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5527.8     | 4094.642 | 4.095  | 5298.0     | 5518.0        |

Final op time: 4.144 us (5594.6 cycles), device=5, bucket=NCRISC transport

