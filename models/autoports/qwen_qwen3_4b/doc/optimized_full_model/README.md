# Qwen3-4B Optimized Full Model

Stage: optimized-full-model
Model: Qwen/Qwen3-4B
Target: 1x4 Blackhole p150b ring, TP4, `FABRIC_1D_RING`
Status: implementation and evidence complete; `$stage-review` rereview returned `clean-pass`.

## Current Performance

| Path | Before | Current | Evidence |
| --- | ---: | ---: | --- |
| AIME24 teacher-forcing traced decode | TTFT `846.76 ms`, decode `36.66 t/s/u` | TTFT `837.89 ms`, decode `63.95 t/s/u`, top5/top100 `100/100` | `logs/run_teacher_forcing_batched_kernel.log` |
| Token-out traced decode, no per-token readback, full path including prepare/capture | TTFT `342.20 ms`, decode `37.77 t/s/u`, readbacks `0` | TTFT `338.98 ms`, decode `41.41 t/s/u`, readbacks `0` | `token_out_no_readback_custom_sampler_batched_kernel_benchmark.json` |
| Token-out traced decode, steady replay after capture | not split out | `96.91 t/s/u` (`10.32 ms/token`), readbacks `0` | `perf_summary.json` |
| Token-out traced decode with per-token readback | TTFT `332.59 ms`, decode `37.69 t/s/u` | superseded by no-readback metric | `../full_model/token_out_timing_trace.log` |

Accuracy refresh:

| Gate | top-1 | top-5 | top-100 | Evidence |
| --- | ---: | ---: | ---: | --- |
| AIME24 chat-template prefill | `93/100` | `100/100` | `100/100` | `logs/run_prefill_check_custom_sampler_common_args.log` |
| AIME24 teacher forcing | `94/100` | `100/100` | `100/100` | `logs/run_teacher_forcing_batched_kernel.log` |

## Accepted Runtime Contract

- Full-model TP4 path keeps embeddings, all 36 decoder layers, final RMSNorm, vocab-sharded LM head, on-device greedy sampling, persistent token feedback, page-table/cache state, and generator orchestration in the optimized path.
- Greedy token-out decode uses a Qwen-local TP4 sampler: per-device tile-local argmax over shard-local logits, a tiny all-gather of four `(score, token_id)` pairs, and a pair reducer that writes replicated `tt_out_tok` for next-token feedback. No full-vocab all-gather, force-argmax, or generic `TopKDeviceOperation` is on the measured greedy path.
- Top-k/top-p-capable sampling remains available separately through `Sampling1D(max_top_k=32)`.
- The selected decoder stack policy is preserved: TP4, BFP4 weights, BF16 activations/KV cache, LoFi math, replicated BF16 layer boundary, persistent CCL path.
- Non-aligned prompt support remains covered by the AIME24 chat-template prompt and contract tests.
- Serving state is explicit: cache, page table, prompt lengths, current positions, RoPE positions, token feedback, fixed slots, and page-table generation are tracked by the generator/model trace state.
- Mixed prompt lengths and fixed slots were smoke-tested with `max_batch_size=4`; due the current TTNN paged-cache/page-table batch contract, inactive slots are carried as dummy rows and ignored by the scheduler rather than omitted from the traced batch.
- `doc/context_contract.json` remains full HF context: target `40960`, supported `40960`.

## Perf Report

Final reduced full-model Tracy capture uses one real decoder layer plus the real embedding, final norm, LM head, greedy sampler, cache/page-table state, token feedback, and trace replay path.

Final batched-kernel artifacts:

- `tracy/reduced_full_model_custom_sampler_batched_kernel/tt_perf_report_token_out_steady.txt`
- `tracy/reduced_full_model_custom_sampler_batched_kernel/tt_perf_report_token_out_steady.csv`
- `tracy/reduced_full_model_custom_sampler_batched_kernel/tt_perf_report_token_out_steady_stacked.csv`
- `tracy/reduced_full_model_custom_sampler_batched_kernel/tt_perf_report_token_out_steady_by_op_stacked.csv`
- `tracy/reduced_full_model_custom_sampler_batched_kernel/reports/qwen3_4b_custom_sampler_batched_kernel_reduced_steady/2026_07_02_22_44_51/ops_perf_results_qwen3_4b_custom_sampler_batched_kernel_reduced_steady_2026_07_02_22_44_51.csv.gz`

Steady replay by-op summary:

| Op group | Device time over reduced steady window | Rows | Conclusion |
| --- | ---: | ---: | --- |
| `MatmulDeviceOperation` | `1701.49 us` | 36 | LM head remains the largest terminal matmul component, but no full-vocab all-gather is present. |
| `GenericOpDeviceOperation` | `1025.59 us` | 12 | Custom local-winner/reducer work; no generic TopK row remains. |
| `LayerNormDeviceOperation` | `463.92 us` | 30 | Final/layer norms are below LM-head and sampler work. |
| `AllReduceAsyncDeviceOperation` | `188.29 us` | 12 | Decoder CCL is not the largest avoidable gap. |
| `PagedUpdateCacheDeviceOperation` | `44.17 us` | 12 | Cache update cost is small in the steady traced path. |

`TopKDeviceOperation` rows: `0`.

Optimized multichip decoder layer lower bound from `../optimized_multichip_decoder/README.md`: traced decode `0.279089 ms/layer`, so a 36-layer stack lower bound is `10.047 ms/token` before terminal work. Current steady token-out no-readback wall time is `10.319 ms/token`, about `2.7%` over the stack-only lower bound and below the stack plus reduced terminal-work estimate (`10.047 + 0.690 ms/token`). The older full-path decode number that includes trace prepare/capture remains reported separately as `41.41 t/s/u`.

## Qualitative

Prompt-correct qualitative artifacts were refreshed under `qualitative/`:

- `qualitative_prompt_format.json`: Qwen chat template present, prompt mode `chat`, rendered with `tokenizer.apply_chat_template(..., add_generation_prompt=True)`.
- `qualitative_rendered_prompts.json`: six shared prompts from `models/common/readiness_check/vllm_prompts.txt`.
- `vllm_qualitative_outputs.json`: HF greedy controls and TT greedy outputs.
- `qualitative_verdict.json`: all six prompts pass.

Free-running autoregressive output was refreshed in `autoregressive/`, with 100 HF tokens and 100 TT tokens. The shared degenerate-output checker reports `No degenerate output detected` in `logs/check_degenerate_output_common_args.log`.

## Known Limitations

- The no-readback greedy path is optimized for TP4 `1x4` P150. Other mesh shapes intentionally raise until a matching cross-device reducer is implemented.
- Watcher ETH instrumentation remains disabled for watcher evidence because ACTIVE_ETH fabric startup exceeds the watcher kernel config buffer on this checkout; the worker-kernel model path is watcher-clean with `TT_METAL_WATCHER_DISABLE_ETH=1`.
- Sparse inactive rows are represented as dummy rows in fixed-slot token-out traces. A physically sparse active-row trace would require TTNN paged-cache support for page-table batch larger than input batch.
