# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_22_53
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

Report: generated/profiler/reports/2026_11_02_09_23_06/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.6       | 55.259   | 0.055  | 72.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6727.1     | 4983.062 | 4.983  | 6639.0     | 6730.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6849.2     | 5073.457 | 5.073  | 6696.0     | 6820.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6690.0     | 4955.531 | 4.956  | 4238.0     | 6768.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6678.4     | 4946.988 | 4.947  | 2751.0     | 6798.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6717.0     | 4975.580 | 4.976  | 6252.0     | 6734.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6855.1     | 5077.827 | 5.078  | 6664.0     | 6827.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6705.5     | 4967.037 | 4.967  | 4680.0     | 6769.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 6698.8     | 4962.099 | 4.962  | 3419.0     | 6800.5        |

Final op time: 5.078 us (6855.1 cycles), device=5, bucket=NCRISC transport

