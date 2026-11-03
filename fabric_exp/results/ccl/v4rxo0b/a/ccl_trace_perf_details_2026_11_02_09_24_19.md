# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_24_19
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

Report: generated/profiler/reports/2026_11_02_09_24_32/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.2       | 56.420   | 0.056  | 70.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10603.4    | 7854.370 | 7.854  | 10516.0    | 10585.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10644.3    | 7884.642 | 7.885  | 10217.0    | 10649.0       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10494.1    | 7773.407 | 7.773  | 7141.0     | 10607.5       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10461.2    | 7749.012 | 7.749  | 4745.0     | 10649.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10554.7    | 7818.296 | 7.818  | 9467.0     | 10591.0       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10653.3    | 7891.333 | 7.891  | 10432.0    | 10644.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10492.2    | 7772.025 | 7.772  | 7234.0     | 10609.5       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10487.2    | 7768.321 | 7.768  | 5428.0     | 10666.0       |

Final op time: 7.891 us (10653.3 cycles), device=5, bucket=NCRISC transport

