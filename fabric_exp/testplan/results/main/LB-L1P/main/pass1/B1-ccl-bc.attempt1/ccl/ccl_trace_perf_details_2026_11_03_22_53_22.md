# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_03_22_53_22
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

Report: generated/profiler/reports/2026_11_03_22_53_33/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 77.6       | 57.457   | 0.057  | 71.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5508.7     | 4080.519 | 4.081  | 5374.0     | 5499.5        |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5596.9     | 4145.827 | 4.146  | 5398.0     | 5532.0        |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5426.0     | 4019.235 | 4.019  | 4086.0     | 5458.5        |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5469.2     | 4051.259 | 4.051  | 3348.0     | 5533.5        |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5484.5     | 4062.593 | 4.063  | 5242.0     | 5467.0        |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5630.5     | 4170.765 | 4.171  | 5342.0     | 5539.0        |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5448.7     | 4036.074 | 4.036  | 4818.0     | 5449.5        |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 5508.3     | 4080.247 | 4.080  | 4490.0     | 5527.5        |

Final op time: 4.171 us (5630.5 cycles), device=5, bucket=NCRISC transport

