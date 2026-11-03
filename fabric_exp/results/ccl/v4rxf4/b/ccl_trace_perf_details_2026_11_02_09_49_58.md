# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_49_58
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

### num_links=2, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_02_09_50_11/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 76.2       | 56.469   | 0.056  | 70.0       | 78.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10568.3    | 7828.370 | 7.828  | 10457.0    | 10565.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10625.1    | 7870.469 | 7.870  | 10480.0    | 10624.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10458.2    | 7746.790 | 7.747  | 7103.0     | 10565.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10435.4    | 7729.901 | 7.730  | 4781.0     | 10629.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10512.4    | 7786.988 | 7.787  | 9132.0     | 10539.0       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10616.4    | 7864.025 | 7.864  | 10252.0    | 10624.0       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10450.8    | 7741.309 | 7.741  | 6995.0     | 10539.5       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10449.9    | 7740.667 | 7.741  | 5222.0     | 10633.5       |

Final op time: 7.870 us (10625.1 cycles), device=1, bucket=NCRISC transport

