# Optimized Full Model Work Log

Initial SHA: `fbd92bb1c8c33405bb075dccf7e71a4fbce894f7`
Latest local commit: recorded in the final handoff after commit creation.

## Final Checks

```bash
python -m py_compile \
  models/autoports/qwen_qwen3_4b/tt/model.py \
  models/autoports/qwen_qwen3_4b/tt/generator.py \
  models/autoports/qwen_qwen3_4b/tt/optimized_decoder.py \
  models/autoports/qwen_qwen3_4b/tt/multichip_decoder.py

pytest -q \
  models/autoports/qwen_qwen3_4b/tests/test_full_model_contract.py \
  models/autoports/qwen_qwen3_4b/tests/test_optimized_full_model.py \
  --tb=short

python .agents/scripts/check_context_contract.py \
  --model-dir models/autoports/qwen_qwen3_4b
```

Results:

- `py_compile`: pass.
- Pytest: `10 passed, 1 skipped`.
- Context contract: target `40960`, supported `40960`, full HF context.

## Accuracy And Performance

Final AIME24 prefill:

```bash
python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/qwen_qwen3_4b \
  --reference models/autoports/qwen_qwen3_4b/doc/full_model/readiness_aime24_chat.refpt \
  --mesh-device P150_X4 --fabric-config FABRIC_1D_RING \
  2>&1 | tee models/autoports/qwen_qwen3_4b/doc/optimized_full_model/logs/run_prefill_check_custom_sampler_common_args.log
```

Result: top1 `93/100`, top5 `100/100`, top100 `100/100`.

Final AIME24 teacher forcing after the batched sampler/kernel update:

```bash
python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/qwen_qwen3_4b \
  --reference models/autoports/qwen_qwen3_4b/doc/full_model/readiness_aime24_chat.refpt \
  --mesh-device P150_X4 --fabric-config FABRIC_1D_RING \
  2>&1 | tee models/autoports/qwen_qwen3_4b/doc/optimized_full_model/logs/run_teacher_forcing_batched_kernel.log
```

Result: top1 `94/100`, top5 `100/100`, top100 `100/100`, TTFT `837.89 ms`, decode `63.95 t/s/u`, e2e `41.91 t/s/u`.

Token-out no-readback benchmark:

```bash
python - <<'PY'
# build generator, load AIME reference prompt, run
# gen.benchmark_token_out_no_readback(prompt, 100, enable_trace=True)
PY
```

Result in `token_out_no_readback_custom_sampler_batched_kernel_benchmark.json`:

```json
{
  "ttft_ms": 338.98349571973085,
  "decode_t/s/u": 41.40574692024547,
  "prepare_decode_ms": 1379.7581121325493,
  "steady_decode_t/s/u": 96.91318318966302,
  "steady_decode_tokens": 98,
  "e2e_t/s/u": 36.630627793298004,
  "decode_tokens": 99,
  "trace_counters": {
    "trace_replays": 98,
    "token_host_refreshes": 0,
    "position_host_refreshes": 2,
    "page_table_host_refreshes": 0,
    "syncs": 2,
    "readbacks": 0
  }
}
```

## Sampler Debug And Fixes

Focused sampler correctness debug found that the first custom sampler read BF16 standard tiles as row-major. Row 0 columns 16-31 live in face 1 at offsets `256..271`, not offsets `16..31`. After the face-offset fix, traced decode steps 1-23 matched same-logits host argmax and the full teacher-forcing gate returned top5/top100 `100/100`.

Watcher then found unique runtime-arg access in the custom kernels. The production tile-local and pair-reduce kernels, plus the local probe kernel, now use common runtime args. The worker-kernel model path is watcher-clean with ETH watcher disabled:

```bash
TT_METAL_WATCHER=10 \
TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DISABLE_ETH=1 \
python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/qwen_qwen3_4b \
  --reference models/autoports/qwen_qwen3_4b/doc/full_model/readiness_aime24_chat.refpt \
  --mesh-device P150_X4 --fabric-config FABRIC_1D_RING \
  2>&1 | tee models/autoports/qwen_qwen3_4b/doc/optimized_full_model/logs/run_teacher_forcing_custom_sampler_common_args_watcher_disable_eth.log
```

Result: pass, top5/top100 `100/100`, no watcher assert. The ETH-enabled watcher attempt failed during ACTIVE_ETH fabric startup before model code with a watcher kernel config-buffer size limit.

## Batched State And Fixed Slots

Stage review found the first optimized token-out path was effectively batch-1 only. The fix made custom greedy sampling batch-aware:

- `Qwen3GreedyTP4Sampler` allocates local/gather pair buffers for `max_batch_size`.
- Tile-local winner and pair-reduce kernels take `active_batch_size` and reduce each active row independently.
- `prepare_token_out_decode` accepts scalar or batched token/position/prompt-length state.
- Sampling trace cache keys include active batch size.
- Top-k/top-p sampling parameter tensors are sized from the active token-out tensor, not hard-coded batch 1.

Hardware smoke:

```bash
python - <<'PY' 2>&1 | tee models/autoports/qwen_qwen3_4b/doc/optimized_full_model/logs/batched_token_out_fixed_slots_smoke.log
# one-layer generator, max_batch_size=4, mixed prompt lengths
# rows 0-1 real, rows 2-3 dummy inactive slots carried through traced token-out
PY
```

Result: `FIXED_SLOT_ROWS 4`, `FIXED_SLOT_REAL_ROWS 2`, `FIXED_SLOT_INACTIVE_DUMMY_ROWS [2, 3]`, counters `token_host_refreshes=0`, `page_table_host_refreshes=0`.

Attempted sparse inactive-slot representations found two current TTNN physical limits: `max_num_blocks=1280` must divide evenly across page-table batch, and `paged_update_cache` requires page-table batch equal to input batch. The accepted representation keeps fixed slots as full traced rows and lets the scheduler ignore inactive rows.

## Perf Report

Final reduced Tracy:

```bash
QWEN3_4B_FULL_MODEL_RUN_REDUCED_PERF=1 \
QWEN3_4B_FULL_MODEL_REDUCED_NEW_TOKENS=8 \
QWEN3_4B_FULL_MODEL_REDUCED_PERF_OUT=models/autoports/qwen_qwen3_4b/doc/optimized_full_model/reduced_full_model_custom_sampler_batched_kernel \
python -m tracy -r -p -v --sync-host-device --dump-device-data-mid-run --check-exit-code \
  -o models/autoports/qwen_qwen3_4b/doc/optimized_full_model/tracy/reduced_full_model_custom_sampler_batched_kernel \
  -n qwen3_4b_custom_sampler_batched_kernel_reduced_steady \
  -m pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_optimized_full_model.py::test_reduced_full_model_token_out_perf_signposts --tb=short \
  2>&1 | tee models/autoports/qwen_qwen3_4b/doc/optimized_full_model/logs/reduced_full_model_custom_sampler_batched_kernel_tracy.log
```

Result: `1 passed`.

Rendered artifacts:

- `tracy/reduced_full_model_custom_sampler_batched_kernel/tt_perf_report_token_out_steady.txt`
- `tracy/reduced_full_model_custom_sampler_batched_kernel/tt_perf_report_token_out_steady.csv`
- `tracy/reduced_full_model_custom_sampler_batched_kernel/tt_perf_report_token_out_steady_stacked.csv`
- `tracy/reduced_full_model_custom_sampler_batched_kernel/tt_perf_report_token_out_steady_by_op_stacked.csv`
- `tracy/reduced_full_model_custom_sampler_batched_kernel/reports/qwen3_4b_custom_sampler_batched_kernel_reduced_steady/2026_07_02_22_44_51/ops_perf_results_qwen3_4b_custom_sampler_batched_kernel_reduced_steady_2026_07_02_22_44_51.csv.gz`

Conclusion: `TopKDeviceOperation` disappeared from the measured greedy token-out path. Final steady summary:

- `MatmulDeviceOperation`: `1701.49 us`, 36 rows.
- `GenericOpDeviceOperation`: `1025.59 us`, 12 rows, custom local-winner/reducer.
- `LayerNormDeviceOperation`: `463.92 us`, 30 rows.
- `AllReduceAsyncDeviceOperation`: `188.29 us`, 12 rows.
- TopK rows: `0`.

Decoder stack lower bound is `0.279089 ms/layer * 36 = 10.047 ms/token`. Final steady token-out is `10.319 ms/token`, about `2.7%` slower than the stack-only lower bound and below the stack plus reduced terminal-work estimate (`10.047 + 0.690 ms/token`).

## Qualitative

Prompt-correct qualitative suite:

```bash
python - <<'PY' 2>&1 | tee models/autoports/qwen_qwen3_4b/doc/optimized_full_model/logs/qualitative_chat_suite_common_args.log
# six prompts from models/common/readiness_check/vllm_prompts.txt
# render with tokenizer.apply_chat_template(..., add_generation_prompt=True)
# run HF greedy controls and TT greedy outputs, max_new_tokens=64
PY
```

Artifacts:

- `qualitative/qualitative_prompt_format.json`
- `qualitative/qualitative_rendered_prompts.json`
- `qualitative/vllm_qualitative_outputs.json`
- `qualitative/qualitative_verdict.json`

Result: all six prompts pass. Tokenizer class `Qwen2Tokenizer`, chat template present, prompt mode `chat`.

Free-running autoregressive:

```bash
python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/qwen_qwen3_4b \
  --hf-model Qwen/Qwen3-4B \
  --prompt-file models/common/readiness_check/autoregressive_prompt.txt \
  --mesh-device P150_X4 --fabric-config FABRIC_1D_RING \
  --output-dir models/autoports/qwen_qwen3_4b/doc/optimized_full_model/autoregressive \
  --max-new-tokens 100 \
  2>&1 | tee models/autoports/qwen_qwen3_4b/doc/optimized_full_model/logs/run_autoregressive_common_args.log
```

Result: HF and TT each produced 100 tokens. TT stayed coherent in English.

Degenerate-output audit:

```bash
python models/common/readiness_check/check_degenerate_output.py \
  --model-dir models/autoports/qwen_qwen3_4b \
  --scope all \
  --missing-artifacts advisory \
  --json models/autoports/qwen_qwen3_4b/doc/optimized_full_model/degenerate_output_report.json \
  2>&1 | tee models/autoports/qwen_qwen3_4b/doc/optimized_full_model/logs/check_degenerate_output_common_args.log
```

Result: `No degenerate output detected`, including the optimized prompt-suite `vllm_qualitative_outputs.json` and optimized autoregressive artifact.

## Accepted Changes

- Added low-level traced token-out APIs: `prepare_token_out_decode`, `decode_next_token_traced`, `refresh_decode_page_table`, and `benchmark_token_out_no_readback`.
- Added changed-only decode trace page-table refresh through `Qwen3FullModel.refresh_trace_page_table`.
- Added explicit `max_batch_size` generator state and reset behavior for fixed-slot serving preparation.
- Split `greedy_sampler`, `topk_sampler`, and `force_argmax_sampler` so optimized greedy avoids generic top-k while top-k/top-p capability remains available.
- Added `Qwen3GreedyTP4Sampler` using `qwen_argmax_tile_local_winner_kernel.cpp`, a tiny TP4 all-gather of local pairs, and `qwen_argmax_pair_reduce_kernel.cpp`.
- Fixed BF16 standard-tile indexing in the local-winner kernel.
- Added gated reduced full-model perf test with outer and steady token-out signposts.

## Rejected Or Refuted Candidates

- LM-head explicit output/program config: rejected due sampler-ready shape failure and later L1-sharded LM-head stall.
- Greedy `Sampling1D` `max_top_k=1`, `max_top_k=8`, and padded top-k variants: rejected due shape failures or timeouts.
- Python-level local-top1 greedy composition using row-major argmax/max plus tiny all-gather: rejected due synthetic timeouts.
- TopK UInt16/multicore predicate patch: rejected and reverted after exact `Sampling1D._topk` probes timed out.
- DeepSeek B1 custom sampler direct wrapper: retained as repro for BRISC/TRISC launch hang, not accepted. Reader-only local reduction primitive did succeed and informed the Qwen-local sampler.
- Force-argmax: rejected for the measured optimized path; prior evidence produced incorrect token choices.

## Stage Review

First `$stage-review` verdict: `more-work-needed`.

Findings fixed:

- Batch-1-only token-out: fixed with batched sampler buffers, batched kernels, batched token-out input normalization, and fixed-slot smoke evidence.
- Lower-bound closure unsupported by old headline: fixed by splitting prepare/capture from steady replay and documenting steady `96.91 t/s/u` versus the decoder stack lower bound.
- Missing `perf_summary.json` and malformed text report: fixed with `perf_summary.json` and regenerated final text/CSV `tt-perf-report` artifacts.

Rereview verdict: `clean-pass`.

The rereview confirmed the batch/fixed-slot token-out fix, lower-bound closure split, `perf_summary.json`, and regenerated parseable text/CSV `tt-perf-report` artifacts. It found no blocking residual work. The only residual note was the documented TTNN paged-cache physical limit for sparse inactive rows; fixed slots are represented as dummy rows and ignored by the scheduler.

## Device Recovery Notes

- After repeated killed full-model sampler probes, `tt-smi -r all --no_reinit` was run once. Post-reset telemetry was clean and later mesh-open/full-model checks succeeded.
- After the watcher caught the runtime-arg bug, `tt-smi -r` recovered the system and a mesh-open smoke passed.
- Hardware-facing commands were serialized after resets; watcher and profiler runs remain separated.

## Handoff

Stage-owned changes are committed locally; the exact final commit SHA is recorded in the handoff message because the SHA is only known after commit creation.
