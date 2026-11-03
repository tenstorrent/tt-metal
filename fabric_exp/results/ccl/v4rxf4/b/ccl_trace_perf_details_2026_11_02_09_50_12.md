# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_50_12
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

Report: generated/profiler/reports/2026_11_02_09_50_24/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.8       | 56.864   | 0.057  | 70.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10603.0    | 7854.099 | 7.854  | 10510.0    | 10606.0       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10618.4    | 7865.481 | 7.865  | 9469.0     | 10643.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10509.9    | 7785.111 | 7.785  | 7662.0     | 10611.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10498.9    | 7776.938 | 7.777  | 5881.0     | 10653.5       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10565.6    | 7826.395 | 7.826  | 9736.0     | 10591.0       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10639.1    | 7880.790 | 7.881  | 9989.0     | 10667.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10534.9    | 7803.605 | 7.804  | 8175.0     | 10621.0       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10535.2    | 7803.852 | 7.804  | 6803.0     | 10680.0       |

Final op time: 7.881 us (10639.1 cycles), device=5, bucket=NCRISC transport

