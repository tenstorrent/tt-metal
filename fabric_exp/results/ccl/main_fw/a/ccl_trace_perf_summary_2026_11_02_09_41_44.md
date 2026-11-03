# DeepSeek CCL trace perf matrix

Timestamp: 2026_11_02_09_41_44
CCLs: sdpa_reduce_to_all
Details file: fabric_exp/results/ccl/main_fw/a/ccl_trace_perf_details_2026_11_02_09_41_44.md

## sdpa_reduce_to_all

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_sdpa_reduce_to_all.py::test_sdpa_reduce_to_all_trace
Configured benchmark shape: [8, 4096]
Configured transport mode: fixed forwarder topology
Setup: 2 forwarder cores total; each forwarder core binds to one physical link, with BRISC/NCRISC handling the two traffic directions on that core.
Configured num_l_chunks: 2
Configured compute_block_sizes: 4
Sweep: num_l_chunks x compute_block_size x max_payload_size_bytes
For this shape, total_l_tiles per worker = 16.
Tuned payload matrix: 2048 -> (4, 8), 4096 -> (2, 4), 8192 -> (2, 8), 15232 -> (2, 8).
Invalid cells are shown as n/a when a forced L chunk payload exceeds the configured fabric max payload or a compute block size does not divide the total L tiles.
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Conversion: us = cycles / CHIP_FREQ[MHz], ns = cycles * 1000 / CHIP_FREQ[MHz].
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).
Top-level zones: SDPA_REDUCE_READER, SDPA_REDUCE_WRITER, SDPA_REDUCE_COMPUTE, SDPA_REDUCE_FORWARDER

Observed CHIP_FREQ[MHz]: 1350.000

### Per-run overview

| num_l_chunks | tiles_per_l_chunk | block_size | max_payload_size_bytes | op_time_us | op_cycles | op_bucket       |
| ------------ | ----------------- | ---------- | ---------------------- | ---------- | --------- | --------------- |
| 2            | 8                 | 4          | 4096                   | 5.615      | 7579.8    | TRISC_2 compute |

### Perf matrix (op time, us)

| num_l_chunks, block_size \ max_payload | 4096  |
| -------------------------------------- | ----- |
| 2, 4                                   | 5.615 |

### Perf matrix (op time, cycles)

| num_l_chunks, block_size \ max_payload | 4096   |
| -------------------------------------- | ------ |
| 2, 4                                   | 7579.8 |

