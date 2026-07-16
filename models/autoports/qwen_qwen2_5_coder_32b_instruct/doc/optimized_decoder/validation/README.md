# Retained optimized-decoder validation

These are primary command outputs from the final implementation and its
material alternatives. All hardware commands used the repo's `python_env`,
the real checkpoint snapshot recorded in `../work_log.md`, one Blackhole
device, and no `TT_VISIBLE_DEVICES` override.

## Correctness and watcher

| Artifact | Command/test | What it proves |
| --- | --- | --- |
| `watcher_correctness.log` | `TT_METAL_WATCHER=10 ... pytest tests/test_optimized_decoder.py -q -s` | final default QKV, non-aligned seq 17/33, batch-1 repeated decode/determinism, and batch-32 real-layer PCC; 5 passed, 2 opt-in skips |
| `watcher_device.log` | device watcher dump copied from that exact run | watcher-clean completion without device assert, hang, timeout, illegal access, or corruption |
| `candidate_sweep.log` | each listed `QWEN2_5_CODER_32B_OPT_PROFILE` with `test_real_layer32_profile_matches_functional_bar` | real-layer precision/fidelity and 16/32-core PCC accept/reject evidence; expected threshold failures are retained verbatim |
| `packed40_batch32_correctness.log` | profile `packed_mlp_bfp8_hifi2_dram_gate40c` with `test_default_batch32_advisor_real_layer32_matches_functional_bar` | strongest prior timing candidate is also correct at batch 32: prefill 0.999214, decode 0.999361 |

## Paired traced timing

Every timing log uses
`test_warmed_prefill_and_traced_decode_perf`, 50 trace replays, seq-17
prefill, and records the complete machine-readable `PERF_RESULT`. The test
also checks finite output, bitwise same-input replay, refreshed-input
observability, and that readbacks stay outside the timing window.

| Artifact | Batch/profile | Prefill / traced decode mean |
| --- | --- | ---: |
| `functional_batch1_perf.log` | 1 / functional BF16 | 3.1169 / 2.9028 ms |
| `optimized_batch1_perf.log` | 1 / final packed 40-core default | 2.9489 / 2.1503 ms |
| `bfp8_hifi2_dram_16c_batch1_perf.log` | 1 / correct split 16-core | 2.9485 / 2.1929 ms |
| `packed_mlp_bfp8_hifi2_dram_32c_batch1_perf.log` | 1 / correct packed 32-core | 3.0088 / 2.1519 ms |
| `functional_perf.log` | 32 / functional BF16 | 83.2726 / 82.3735 ms |
| `bfp8_hifi2_dram_32c_batch32_perf.log` | 32 / correct split 32-core | 9.9351 / 2.3240 ms |
| `packed_mlp_bfp8_hifi2_dram_gate40c_batch32_perf.log` | 32 / correct packed 40-core | 9.9815 / 2.2906 ms |
| `optimized_perf.log` | 32 / final advisor packed default | 9.9975 / 1.9410 ms |

The common timing command is:

```bash
QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=<snapshot> \
QWEN2_5_CODER_32B_OPT_PERF=1 \
QWEN2_5_CODER_32B_PERF_DECODER=<functional-or-optimized> \
QWEN2_5_CODER_32B_PERF_BATCH=<1-or-32> \
QWEN2_5_CODER_32B_PERF_REPS=50 \
QWEN2_5_CODER_32B_OPT_PROFILE=<candidate-when-applicable> \
python_env/bin/pytest \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_optimized_decoder.py::test_warmed_prefill_and_traced_decode_perf -q -s
```

## Program-geometry rejection logs

`geometry/gate_in0_block{4,8,10,16}.log` are continued, role-isolated trials
showing that the winning gate/up input shard is two tiles wide and therefore
cannot use those blocks. `geometry/down_in0_block{5,10}.log` show that total
tiled down K=864 is not divisible by either non-power candidate. These trials
followed the initial whole-graph error, so a first TTNN/API error was not used
as the rejection evidence.

Compact Tracy/`tt-perf-report` artifacts and their captured console output are
in `../tracy/layer32/`; the profiler command and cumulative contract are in
`../work_log.md`.
