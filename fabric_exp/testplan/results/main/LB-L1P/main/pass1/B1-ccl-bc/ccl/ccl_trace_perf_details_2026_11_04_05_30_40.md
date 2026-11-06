# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_30_40
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

### num_links=2, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_04_05_30_50/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 77.2       | 57.210   | 0.057  | 71.0       | 77.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10761.5    | 7971.457 | 7.971  | 10598.0    | 10745.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10769.8    | 7977.605 | 7.978  | 9782.0     | 10794.0       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10628.1    | 7872.691 | 7.873  | 7100.0     | 10761.5       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10598.3    | 7850.568 | 7.851  | 4683.0     | 10820.5       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10694.7    | 7921.975 | 7.922  | 9645.0     | 10691.5       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10786.2    | 7989.802 | 7.990  | 10285.0    | 10748.5       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10636.7    | 7879.062 | 7.879  | 7150.0     | 10766.5       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10614.8    | 7862.815 | 7.863  | 5222.0     | 10774.5       |

Final op time: 7.990 us (10786.2 cycles), device=5, bucket=NCRISC transport

