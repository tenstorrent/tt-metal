# DeepSeek CCL trace perf matrix

Timestamp: 2026_11_02_09_52_48
CCLs: all_gather
Details file: fabric_exp/results/ccl/v4rxo0b/b/ccl_trace_perf_details_2026_11_02_09_52_48.md

## all_gather

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_gather.py::test_ccl_all_gather
Configured benchmark shape: [1, 7168]
Configured transport mode: single-link only
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Conversion: us = cycles / CHIP_FREQ[MHz], ns = cycles * 1000 / CHIP_FREQ[MHz].
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).
Top-level zones: ALLGATHER_GATHER, ALLGATHER_TRANSPORT

Observed CHIP_FREQ[MHz]: 1350.000

### Per-run overview

| num_links | max_payload_size_bytes | op_time_us | op_cycles | op_bucket     |
| --------- | ---------------------- | ---------- | --------- | ------------- |
| 1         | 8192                   | 2.546      | 3437.3    | NCRISC gather |

### Perf matrix (op time, us)

| num_links \ max_payload | 8192  |
| ----------------------- | ----- |
| 1                       | 2.546 |

### Perf matrix (op time, cycles)

| num_links \ max_payload | 8192   |
| ----------------------- | ------ |
| 1                       | 3437.3 |

