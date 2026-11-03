# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_37_40
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

### num_links=1, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_02_09_37_54/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.4       | 55.136   | 0.055  | 70.0       | 74.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5524.0     | 4091.852 | 4.092  | 5454.0     | 5513.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5649.1     | 4184.494 | 4.184  | 5522.0     | 5606.5        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5475.3     | 4055.778 | 4.056  | 3435.0     | 5531.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5494.2     | 4069.802 | 4.070  | 2318.0     | 5601.0        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5522.1     | 4090.444 | 4.090  | 5172.0     | 5531.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5669.4     | 4199.556 | 4.200  | 5513.0     | 5609.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5489.1     | 4066.025 | 4.066  | 3946.0     | 5552.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5525.2     | 4092.765 | 4.093  | 3406.0     | 5605.5        |

Final op time: 4.200 us (5669.4 cycles), device=5, bucket=NCRISC transport

