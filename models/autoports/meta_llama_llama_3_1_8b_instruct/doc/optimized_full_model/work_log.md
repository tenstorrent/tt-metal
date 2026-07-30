# Optimized Full Model Work Log

Date: 2026-06-15.

Status: complete for optimized-full-model. The full model has refreshed
accuracy, no-readback token-out trace evidence, reduced Tracy/tt-perf-report
artifacts, lower-bound attribution, sampler-strategy evidence, and a clean
watcher run for the measured token-out path.

## Scope

Used the `$multichip` and `$optimize` guidance for the
`meta-llama/Llama-3.1-8B-Instruct` full model. vLLM integration was not started.

The implementation preserves the completed full-model path:

- `tt/model.py` full model with embeddings, 32 optimized multichip decoder
  layers, final norm, split BF8 LM head, and paged KV cache.
- `tt/generator.py` readiness generator with prefill, traced decode, split
  sampling, `tt_out_tok` feedback, persistent token/position/RoPE/page-table
  inputs, and device-side position/RoPE advance.
- Decoder policy
  `llama31_8b_t3k_1x8_tp8_bfp4_attn_bfp4_mlp_bfp8_act_decode_v2`.

## Code Changes

Updated `tt/generator.py`:

- Added `readback: bool = True` to `_decode_trace_sample`.
- When `readback=False`, the model and sampling traces replay with `tt_out_tok`
  feedback but do not read the sampled token back to host.
- Added `_release_decode_trace`.
- Made `reset()` release model/sampling decode traces by default; callers can
  pass `keep_decode_trace=True` for explicit trace-reuse tests.
- Made `teardown()` release decode traces.

Updated `doc/full_model/token_out_trace_evidence.py`:

- Added no-readback token-out measurement for the serving-style path.
- Made the default evidence run measure no-readback token-out plus trace
  feedback and top-k/top-p smoke.
- Added `--token-out-only` for scoped watcher/runtime-integrity runs of the
  measured no-readback path after the full trace-feedback/top-k probes already
  have normal evidence.
- Kept readback-inclusive token-out disabled by default because same-process
  readback plus no-readback attempts can break later prefill trace/buffer
  lifetime.

Added `doc/full_model/sampling_strategy_benchmark.py`:

- Benchmarks traced split greedy and split top-k/top-p sampling on real
  full-model logits.
- Records that force-argmax is not exposed by this model's sampling args and is
  not selected for greedy completion evidence.

Updated `doc/full_model/reduced_profile.py`:

- Changed the profiled token-out replay window to use `readback=False`, matching
  the optimized no-readback serving path.

## Commands And Results

Hardware discovery:

```bash
tt-smi -ls --local
```

Result: chips `0-7` visible on T3K.

Static compile:

```bash
python_env/bin/python -m py_compile \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tt/generator.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/token_out_trace_evidence.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/reduced_profile.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/sampling_strategy_benchmark.py
```

Result: passed.

No-readback token-out evidence:

```bash
timeout 7200s python_env/bin/python -u models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/token_out_trace_evidence.py \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --prompt-file models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/prompt_128.txt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  --max-new-tokens 128 \
  --output-json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/token_out_trace_evidence.json \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/token_out_trace_evidence_stdout.txt
```

Result: passed.

- TTFT `629.731884 ms`.
- First decode step with model and sampling trace capture `536.401989 ms`.
- Steady replay `70.575829 t/s/u`, `14.169157 ms/token`.
- Sampled-token readbacks `0`.
- Token/position/RoPE/page-table host copies `1/1/1/1`.
- Position/RoPE device increments `127/127`.
- Greedy uses split sampling, `force_argmax=False`, `max_top_k=32`.
- Trace feedback, changed-only page-table, and top-k/top-p smoke assertions
  passed.

Sampler strategy benchmark:

```bash
timeout 7200s python_env/bin/python -u models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/sampling_strategy_benchmark.py \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --prompt-file models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/prompt_128.txt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  --replay-count 64 \
  --output-json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/sampling_strategy_benchmark.json \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/runs/sampling_strategy_benchmark.log
```

Result: passed.

- Split greedy: `0.514226 ms/replay`; sampled token `11` matched row-0 argmax
  `11`; `force_argmax=False`.
- Split top-k/top-p: `0.514234 ms/replay`; same traced split path,
  `force_argmax=False`.
- Force-argmax: unavailable because `make_sampling_args().model_config` has no
  `SAMPLING_AG_CONFIG.allow_force_argmax`; not benchmarked and not selected.

Refreshed AIME24 reference:

```bash
timeout 7200s python_env/bin/python -u -m models.common.readiness_check.generate \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --prompt-source aime24 \
  --chat-template \
  --gen-len 100 \
  --top-k 100 \
  --output models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/aime24_chat_template_100_top100.refpt \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/runs/generate_aime24_chat_template_100_top100.log
```

Result: generated 100 continuation tokens, prompt length 184.

Prefill check:

```bash
timeout 7200s python_env/bin/python -u -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --reference models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/aime24_chat_template_100_top100.refpt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/runs/run_prefill_check_aime24.log
```

Result: top1 `90/100`, top5 `100/100`, top100 `100/100`.

Teacher forcing:

```bash
timeout 7200s python_env/bin/python -u -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --reference models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/aime24_chat_template_100_top100.refpt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/runs/run_teacher_forcing_aime24.log
```

Result: top1 `92/100`, top5 `100/100`, top100 `100/100`,
TTFT `648.31 ms`, decode `49.31 t/s/u`, e2e `37.65 t/s/u`.

Autoregressive:

```bash
timeout 7200s python_env/bin/python -u -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  --max-new-tokens 128 \
  --output-dir models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/autoregressive_story_128 \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/runs/run_autoregressive_story_128.log
```

Result: HF and TT each produced 128 tokens.

Degeneracy check:

```bash
python_env/bin/python models/common/readiness_check/check_degenerate_output.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/autoregressive_story_128 \
  --scope autoregressive \
  --missing-artifacts critical \
  --json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/autoregressive_degenerate_report.json
```

Result: no findings, adjacent duplication `0.0`, trigram loop fraction
`0.0297`, HF/TT token agreement `30/128`.

Reduced Tracy profile:

```bash
timeout 7200s python_env/bin/python -m tracy -r -p -v --check-exit-code \
  -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/tracy/reduced_profile/.logs \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/reduced_profile.py \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --prompt-file models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/prompt_128.txt \
  --output-json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/reduced_profile_summary.json \
  --decode-replays 4
```

Result: passed. Ops CSV generated under
`tracy/reduced_profile/.logs/reports/2026_06_15_17_38_27/`.

Stable report generation:

```bash
cp tracy/reduced_profile/.logs/reports/2026_06_15_17_38_27/ops_perf_results_2026_06_15_17_38_27.csv \
  tracy/reduced_profile/reduced_profile_ops_perf_results.csv

tt-perf-report tracy/reduced_profile/reduced_profile_ops_perf_results.csv \
  --start-signpost PERF_REDUCED_PREFILL \
  --end-signpost PERF_REDUCED_PREFILL_END \
  --no-color \
  --csv tracy/reduced_profile/reduced_prefill_perf_report.csv \
  --summary-file tracy/reduced_profile/reduced_prefill_perf_report_stacked \
  > tracy/reduced_profile/reduced_prefill_perf_report.txt

tt-perf-report tracy/reduced_profile/reduced_profile_ops_perf_results.csv \
  --start-signpost PERF_REDUCED_TOKEN_OUT_DECODE \
  --end-signpost PERF_REDUCED_TOKEN_OUT_DECODE_END \
  --no-color \
  --csv tracy/reduced_profile/reduced_token_out_decode_perf_report.csv \
  --summary-file tracy/reduced_profile/reduced_token_out_decode_perf_report_stacked \
  > tracy/reduced_profile/reduced_token_out_decode_perf_report.txt

tt-perf-report tracy/reduced_profile/reduced_profile_ops_perf_results.csv \
  --start-signpost PERF_REDUCED_TOKEN_OUT_DECODE \
  --end-signpost PERF_REDUCED_TOKEN_OUT_DECODE_END \
  --no-color \
  --no-merge-devices \
  --csv tracy/reduced_profile/reduced_token_out_decode_perf_report_per_device.csv \
  --summary-file tracy/reduced_profile/reduced_token_out_decode_perf_report_per_device_stacked \
  > tracy/reduced_profile/reduced_token_out_decode_perf_report_per_device.txt
```

Result: report CSVs and stacked CSV/PNG artifacts written.

Reduced host timings:

- One-layer prefill `85.908955 ms`.
- One-layer token-out decode min/avg `1.551531 / 1.583331 ms`.

Token-out stacked report top buckets:

- Width-sharded matmuls: `544.46 us`, `39.35%`.
- `TopKDeviceOperation`: `154.23 us`, `11.15%`.
- `AllGatherDeviceOperation`: `133.84 us`, `9.67%`.
- `SamplingDeviceOperation`: `63.81 us`, `4.61%`.
- Width-sharded `AllGatherAsyncDeviceOperation`: `54.84 us`, `3.96%`.
- `ReduceScatterMinimalAsyncDeviceOperation`: `53.54 us`, `3.87%`.

Lower-bound calculation:

- Optimized decoder layer min/avg:
  `0.397455879 / 0.401623140 ms`.
- 32-layer stack lower bound min/avg:
  `12.718588114 / 12.851940468 ms`.
- Reduced one-layer terminal work min/avg:
  `1.154074911 / 1.181707543 ms`.
- Stack plus terminal min/avg:
  `13.872663025 / 14.033648011 ms`.
- Full no-readback token-out:
  `14.169156950 ms/token`.
- Gap versus stack plus terminal:
  `0.296494 ms` (`2.14%`) against min, `0.135509 ms` (`0.97%`) against avg.

## Watcher Recovery And Runtime Audit

Full ETH watcher attempt:

```bash
timeout 7200s env TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
  TT_METAL_LOGS_PATH=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/watcher/token_out_no_readback_128 \
  python_env/bin/python -u models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/token_out_trace_evidence.py \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --prompt-file models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/prompt_128.txt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  --max-new-tokens 128 \
  --output-json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/watcher/token_out_no_readback_128/token_out_trace_evidence_watcher.json \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/watcher/token_out_no_readback_128/watcher_run.log
```

Result: failed after reaching no-readback trace capture/replay. The abort was:

```text
Timeout waiting for Ethernet core service remote IO request.
...
tt::tt_metal::WatcherServer::Impl::poll_watcher_data()
```

`tt-smi -ls --local` after this failure still listed all 8 chips. Output is in
`watcher/token_out_no_readback_128/hardware_status_after_watcher_timeout.log`.

Scoped retry with ETH watcher disabled:

```bash
timeout 7200s env TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_WATCHER_NOINLINE=1 \
  TT_METAL_LOGS_PATH=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/watcher/token_out_no_readback_128_disable_eth \
  python_env/bin/python -u models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/token_out_trace_evidence.py \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --prompt-file models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/prompt_128.txt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  --max-new-tokens 128 \
  --output-json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/watcher/token_out_no_readback_128_disable_eth/token_out_trace_evidence_watcher.json \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/watcher/token_out_no_readback_128_disable_eth/watcher_run.log
```

Result: failed before model execution:

```text
ETH core heartbeat check failed on device ASIC ID: 9956368389, ETH core e7-0
```

`tt-smi -ls --local` after this failure still listed all 8 chips. Output is in
`watcher/token_out_no_readback_128_disable_eth/hardware_status_after_disable_eth_heartbeat_failure.log`.

Hardware recovery:

```bash
tt-smi -r all
tt-smi -ls --local
```

Result: reset PCI devices `0-3`, reinitialized the boards, and listed chips
`0-7`. A minimal T3K `FABRIC_1D_RING` open/close smoke passed.

Full watcher retry after reset:

```bash
timeout 7200s env TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
  TT_METAL_LOGS_PATH=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/watcher/token_out_no_readback_128_after_reset \
  python_env/bin/python -u models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/token_out_trace_evidence.py \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --prompt-file models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/prompt_128.txt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  --max-new-tokens 128 \
  --output-json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/watcher/token_out_no_readback_128_after_reset/token_out_trace_evidence_watcher.json \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/watcher/token_out_no_readback_128_after_reset/watcher_run.log
```

Result: the no-readback token-out path completed and wrote pass JSON, then the
extra trace-feedback/top-k probes hit the same watcher ETH polling timeout.
The pass JSON is retained for provenance, but final runtime signoff uses the
scoped measured-path watcher run below.

Second reset:

```bash
tt-smi -r all
```

Result: reset PCI devices `0-3` again.

Scoped watcher run for the measured optimized path:

```bash
timeout 7200s env TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
  TT_METAL_LOGS_PATH=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/watcher/token_out_only_128_after_reset \
  python_env/bin/python -u models/autoports/meta_llama_llama_3_1_8b_instruct/doc/full_model/token_out_trace_evidence.py \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --prompt-file models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/prompt_128.txt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  --max-new-tokens 128 \
  --token-out-only \
  --output-json models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/watcher/token_out_only_128_after_reset/token_out_trace_evidence_watcher.json \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_full_model/watcher/token_out_only_128_after_reset/watcher_run.log
```

Result: passed with clean watcher shutdown.

- Watcher TTFT `1346.801965 ms`.
- Watcher steady replay `10.319454 t/s/u`, `96.904350 ms/token`.
- Sampled-token readbacks `0`.
- Token/position/RoPE/page-table host copies `1/1/1/1`.
- Greedy uses split sampling, `force_argmax=False`, `max_top_k=32`.
- Watcher detached devices `0-7`; Ethernet retraining events were all `0`.
- Log-only runtime failure signature scan found no matches.
- Runtime fallback audit over the measured token-out and sampler logs found no
  fallback signatures. Artifact: `runtime_fallback_audit.txt`.
- Final `tt-smi -ls --local` after the clean watcher and sampler benchmark
  listed chips `0-7`.

## Completion Audit

- Full path optimized/parallelized: embeddings, RoPE, decoder stack, final norm,
  LM head, split sampling, cache/page table handling, trace replay, collectives,
  residual layouts, and generator orchestration.
- Canonical split-sampling token-out preserved: greedy and top-k/top-p capable,
  `tt_out_tok` feedback, device-side position/RoPE advance, changed-only page
  table probe, and no steady-path sampled-token readback.
- Decoder policy preserved:
  `llama31_8b_t3k_1x8_tp8_bfp4_attn_bfp4_mlp_bfp8_act_decode_v2`.
- AIME24 refreshed: prefill and teacher-forcing both meet top5 `100/100` and
  top100 `100/100`.
- Autoregressive evidence refreshed for 128 tokens with no degeneracy findings.
- Full token-out `14.169157 ms/token` is within `2.14%` of stack-plus-terminal
  lower-bound min and within `0.97%` of the avg estimate.
- `TopKDeviceOperation`, vocab all-gather, and sampling are visible in
  tt-perf-report, but none dominates the measured token-out path.
- Runtime audit for the measured token-out path is clean under watcher.
- No vLLM integration work was started.

## Artifact Index

- `README.md`
- `perf_summary.json`
- `prompt_128.txt`
- `aime24_chat_template_100_top100.refpt`
- `runs/*.log`
- `token_out_trace_evidence.json`
- `token_out_trace_evidence_stdout.txt`
- `runtime_fallback_audit.txt`
- `sampling_strategy_benchmark.json`
- `autoregressive_story_128/*`
- `autoregressive_degenerate_report.json`
- `reduced_profile_summary.json`
- `tracy/reduced_profile/*`
- `watcher/token_out_no_readback_128/*`
- `watcher/token_out_no_readback_128_disable_eth/*`
- `watcher/token_out_no_readback_128_after_reset/*`
- `watcher/token_out_only_128_after_reset/*`
- `hardware_recovery/*`
