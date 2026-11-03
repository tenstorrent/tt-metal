# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_03_02
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

### num_links=2, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_02_10_03_12/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 77.6       | 57.506   | 0.058  | 70.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9473.8     | 7017.605 | 7.018  | 9243.0     | 9489.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9501.4     | 7038.049 | 7.038  | 8399.0     | 9569.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9396.6     | 6960.444 | 6.960  | 6838.0     | 9452.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9397.1     | 6960.815 | 6.961  | 5158.0     | 9602.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9446.8     | 6997.605 | 6.998  | 8757.0     | 9471.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9529.4     | 7058.790 | 7.059  | 9140.0     | 9546.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9409.6     | 6970.074 | 6.970  | 7296.0     | 9469.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 9421.8     | 6979.111 | 6.979  | 6100.0     | 9511.5        |

Final op time: 7.059 us (9529.4 cycles), device=5, bucket=NCRISC transport

