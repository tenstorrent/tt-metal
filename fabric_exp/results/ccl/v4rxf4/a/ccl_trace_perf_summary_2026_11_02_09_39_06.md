# DeepSeek CCL trace perf matrix

Timestamp: 2026_11_02_09_39_06
CCLs: broadcast
Details file: fabric_exp/results/ccl/v4rxf4/a/ccl_trace_perf_details_2026_11_02_09_39_06.md

## broadcast

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_broadcast.py::test_ccl_broadcast
Configured benchmark shape: [1, 7168]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Conversion: us = cycles / CHIP_FREQ[MHz], ns = cycles * 1000 / CHIP_FREQ[MHz].
Op time = slowest device's slowest top-level bucket, using per-device averages across iterations only.
Top-level zones: CCL_BROADCAST_READER, CCL_BROADCAST_TRANSPORT

Cross-device bucket averages are omitted for this CCL.

Observed CHIP_FREQ[MHz]: 1350.000

### Per-run overview

| num_links | max_payload_size_bytes | op_time_us | op_cycles | op_device | op_bucket        |
| --------- | ---------------------- | ---------- | --------- | --------- | ---------------- |
| 2         | 15232                  | 4.115      | 5555.5    | 5         | NCRISC transport |

### Perf matrix (op time, us)

| num_links \ max_payload | 15232 |
| ----------------------- | ----- |
| 2                       | 4.115 |

### Perf matrix (op time, cycles)

| num_links \ max_payload | 15232  |
| ----------------------- | ------ |
| 2                       | 5555.5 |

