# DeepSeek CCL trace perf matrix

Timestamp: 2026_11_04_09_27_52
CCLs: all_reduce
Details file: /data/aagarwal/regress/results-lb2/ab_288afd2_lb2/LB-L1P/main/pass1/B1-ccl-ar/ccl/ccl_trace_perf_details_2026_11_04_09_27_52.md

## all_reduce

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py::test_ccl_all_reduce
Configured benchmark shape: [1, 2048]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 1
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Conversion: us = cycles / CHIP_FREQ[MHz], ns = cycles * 1000 / CHIP_FREQ[MHz].
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).
Top-level zones: CCL_SENDER_WRITER, CCL_RECEIVER, CCL_COMPUTE

Observed CHIP_FREQ[MHz]: 1350.000

### Per-run overview

| num_links | max_payload_size_bytes | op_time_us | op_cycles | op_bucket       |
| --------- | ---------------------- | ---------- | --------- | --------------- |
| 1         | 4352                   | 1.294      | 1747.5    | TRISC_0 compute |

### Perf matrix (op time, us)

| num_links \ max_payload | 4352  |
| ----------------------- | ----- |
| 1                       | 1.294 |

### Perf matrix (op time, cycles)

| num_links \ max_payload | 4352   |
| ----------------------- | ------ |
| 1                       | 1747.5 |

