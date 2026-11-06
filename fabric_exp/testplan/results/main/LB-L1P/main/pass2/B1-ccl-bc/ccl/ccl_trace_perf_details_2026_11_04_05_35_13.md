# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_35_13
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

Report: generated/profiler/reports/2026_11_04_05_35_24/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 74.5       | 55.160   | 0.055  | 72.0       | 72.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5619.0     | 4162.198 | 4.162  | 5493.0     | 5638.0        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5708.2     | 4228.321 | 4.228  | 5505.0     | 5697.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5569.6     | 4125.630 | 4.126  | 3930.0     | 5637.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5581.0     | 4134.049 | 4.134  | 2797.0     | 5685.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5619.4     | 4162.519 | 4.163  | 5379.0     | 5626.5        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5741.1     | 4252.667 | 4.253  | 5437.0     | 5672.5        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5590.9     | 4141.432 | 4.141  | 4472.0     | 5656.0        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5608.5     | 4154.469 | 4.154  | 3498.0     | 5679.5        |

Final op time: 4.253 us (5741.1 cycles), device=5, bucket=NCRISC transport

