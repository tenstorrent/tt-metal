# Decode Sharded Residual/Norm Trial

## Starting Evidence

Stage review flagged that decode still used DRAM-interleaved residual adds and post-attention RMSNorm. The pre-repair current timing row was:

| artifact | traced decode prefix16 |
| --- | ---: |
| `perf_host_timings_current.csv` | 1.283865 ms |
| `perf_host_timings.csv` restored final row | 1.289340 ms |

The pre-repair decode `tt-perf-report` also showed two single-core `LayerNormDeviceOperation` rows around 94 us. The first row is the input RMSNorm before QKV; the second row was the post-attention RMSNorm after the DRAM residual add.

## Candidate Config

The kept candidate uses:

| field | value |
| --- | --- |
| residual memory config | L1 `WIDTH_SHARDED` |
| residual grid | 8x4, 32 cores |
| residual shard shape | `[32, 128]` for batch 1, hidden 4096 |
| RMSNorm program config | `block_h=1`, `block_w=4`, `subblock_w=4`, `inplace=False` |
| post-norm MLP input | kept in the L1 width-sharded residual layout |
| down projection output | existing 8-core L1 width-sharded DRAM-sharded matmul output |
| final residual | add down output to the 32-core sharded attention residual, output in residual layout, then convert final decoder output back to DRAM |

This is not an op-contract blocker. TTNN accepts the sharded add, sharded RMSNorm, sharded MLP input, mixed-shard final add, trace capture, and trace replay.

## A/B Results

Command: a focused prefill-then-traced-decode Python harness using the existing synthetic optimized-decoder test fixtures and monkeypatched decode candidates.

| candidate | status | traced decode host ms | eager decode host ms | synthetic PCC vs HF |
| --- | --- | ---: | ---: | ---: |
| baseline DRAM residual/norm | pass | 1.284928 | 2.742190 | 0.9570590718566394 |
| 8-core sharded residual/post-norm, existing MLP boundary | pass | 1.213185 | 1.995652 | 0.9576700783987007 |
| 32-core sharded residual/post-norm, existing MLP boundary | pass | 1.215583 | 2.242702 | 0.957755904703443 |
| 32-core sharded residual/post-norm, sharded MLP input | pass | 1.130557 | 2.062695 | 0.957683332560469 |
| 32-core sharded residual/post-norm, sharded MLP input and sharded final residual | pass | 1.129039 | 2.140794 | 0.957683332560469 |

Kept: `32-core sharded residual/post-norm, sharded MLP input and sharded final residual`.

Post-implementation perf signpost:

```bash
LLAMA31_8B_OPT_RUN_PERF=1 LLAMA31_8B_OPT_PERF_OUT=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/sharded_residual_norm_perf_host.csv pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_perf_signposts --tb=short
```

Result: `1 passed, 1 warning`; warmed prefill `1.382746 ms`, traced decode `1.130463 ms`.

Tracy-instrumented signpost:

```bash
LLAMA31_8B_OPT_RUN_PERF=1 LLAMA31_8B_OPT_PERF_OUT=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/sharded_residual_norm_perf_host_tracy.csv python -m tracy -r -p -v -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy_sharded_residual_norm -m pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_perf_signposts --tb=short
```

Result: `1 passed, 1 warning`; warmed prefill `1.537558 ms`, traced decode `1.157605 ms`.

Post-change decode `tt-perf-report`:

```bash
tt-perf-report models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy_sharded_residual_norm/optimized_ops.csv --start-signpost PERF_TRACE_DECODE --end-signpost PERF_TRACE_DECODE_END --tracing-mode --no-summary --no-color > models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tt_perf_report_decode_sharded_residual_norm.txt
```

Result: 38 device ops, 0 host ops, 1,074 us device time. The input RMSNorm before QKV remains a 1-core 94 us row. The targeted post-attention RMSNorm moved to a 32-core sharded row at 10 us, with sharded residual add at 2 us, final residual add at 8 us, and final sharded-to-interleaved conversion at 3 us.

## Verification

```bash
python -m py_compile models/autoports/meta_llama_llama_3_1_8b_instruct/tt/optimized_decoder.py models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py
pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_paged_decode_matches_full_hf_layer models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_batched_decode_uses_disjoint_page_rows models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_trace_replay_is_deterministic models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_runtime_has_no_host_fallback --tb=short
```

Result: `5 passed, 1 warning`.

Focused PCC values after implementation:

| check | PCC |
| --- | ---: |
| synthetic paged decode prefix16 | 0.9591494027854894 |
| synthetic paged decode prefix17 | 0.9616032010109328 |
| batch-2 disjoint page rows | 0.9575821607909508 |
| eager-vs-traced decode replay | 1.0 |

## Verdict

Fixed. The decode path now has measured, correct, traced sharded residual/post-norm/MLP-input/final-residual coverage. No TTNN blocker remains for this specific post-attention residual/norm candidate. The remaining 94 us LayerNorm row is the separate input RMSNorm before packed QKV and was not the stage-review target.
