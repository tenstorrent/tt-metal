---
name: vllm-integration
description: Add vLLM serving integration for an existing TTNN full model under the model-specific autoport directory. Use after full-model when producing tt/generator_vllm.py, registering the model with the TT vLLM plugin, running the vLLM readiness server path, and reporting serving-path correctness and performance.
---

# vLLM Integration

## Mission Context

If this skill is used as part of `$model-bringup`, follow that skill's mission, workspace, and reporting contract. This stage starts from a working TTNN full model and generator, then makes it usable through the shared vLLM serving path. The full model itself belongs to `$full-model`; vLLM integration owns the adapter, plugin registration, serving checks, and serving-path performance evidence.

## Your Part

Starting from `models/autoports/<model>/`, add:

```text
tt/generator_vllm.py
```

Expect these files to already exist and pass generator-level readiness:

```text
tt/model.py
tt/generator.py
```

If they do not, use `$full-model` first. During vLLM integration you may make small, evidence-backed changes to `model.py` or `generator.py` when the serving adapter exposes a real contract gap, but avoid turning this stage back into full-model bringup.

Before writing adapter code, load the datatype-sweep selection and confirm the generator constructs that exact policy: weight groups, activation/residual dtype, CCL dtype, KV-cache dtype, compute fidelities, and layer exceptions. Serving uses the selected full-model policy. Serving a reduced-speed model does not satisfy completion.

Load `models/autoports/<model>/doc/context_contract.json` and serve the recorded supported context. The default target is the HF-advertised context. Do not lower `--max-model-len`, model config context, benchmark context, API context, or any other advertised serving capability to work around a model bug. A smaller value is valid only when the context contract records evidence that a hard physical device limit prevents the advertised capability from fitting or running and that the selected value is the largest feasible one.

Serving requests may have any valid prompt length up to that supported context. The adapter must not require prompt length to be divisible by an internal prefill chunk, tile, block, page, or trace size. If the model path pads or chunks internally, the adapter passes the logical prompt length through to masks, positions, cache fill, and output slicing. Include a direct OpenAI-compatible request or targeted runner check with a valid non-aligned prompt length, not only the default 128-token benchmark.

Preserve larger-batch serving capability. The headline performance target is still batch-1 single-user latency, but the adapter must not assume `max_num_seqs=1` or batch size 1 in cache allocation, page tables, scheduler inputs, sampling, async decode, or output formatting. Test serving up to 32 concurrent sequences when the target hardware, memory, and harness allow it. If 32 cannot run, record the largest tested value and the hard physical limit.

## vLLM Adapter

`tt/generator_vllm.py` should delegate to the existing generator's low-level methods. Keep adapter-only code limited to vLLM interface translation.

Read:

- `tech_reports/LLMs/vLLM_integration.md`;
- `models/tt_transformers/tt/generator_vllm.py`;
- the model's existing `tt/generator.py` low-level `prefill_forward` and `decode_forward` methods.

vLLM owns KV-cache allocation in serving mode. Preserve the generator's two cache ownership modes:

- standalone readiness/generator mode, where the generator owns cache allocation and reset;
- vLLM mode, where `allocate_vllm_kv_cache` creates the cache and the adapter passes that exact cache through the low-level API.

Not every cache must be vLLM-owned. vLLM owns the attention KV cache, but recurrent / linear-attention state of constant size (for example conv windows or SSM / gated-delta recurrent state) can live inside the model and be carried across decode steps there, rather than being modeled as a vLLM KV cache. For attention itself, route sliding-window and full-attention layers through vLLM's hybrid attention infrastructure (per-layer KV-cache specs and block tables) instead of forcing a single uniform attention type across all layers.

Make prompt lengths, page tables, decode positions, batch dimensions, trace-side state, and on-device sampling explicit. The serving decode pass must drive the generator's traced decode path, not an eager-only fallback. When adding or debugging trace capture/replay, trace-safe inputs, or replay correctness for this adapter, use `$tt-enable-tracing`. The adapter should not duplicate model logic that already lives in `tt/model.py` or `tt/generator.py`.

For decode performance, implement the vLLM async split before advertising it: `decode_forward(..., read_from_device=False)` should return device tensors, `read_decode_output(..., async_read=True)` should perform the minimal deferred read, and `process_decode_output_host(...)` should do host formatting. Only set `supports_async_decode=True` after this path passes the vLLM plugin's expectations with decode trace enabled and stale-token/current-position tests passing.
When
1. `supports_async_decode=True`
2. sampling on device
2. tracing is enabled
3. reset_batch=False
vLLM may build and submit decode step N+1 before sampled token N has been applied to host scheduler state, so the inputs may be stale or wrong.
This is OK because vLLM expects the model to not read the inputs, and instead have the previous step's sampling output preserved and used as input. Page-table tensor is guaranteed to be unchanged.
Make sure to not read from host in such case, and instead use the inputs already on device.
To allow this, always update the device copy of inputs, such as token/current-position/RoPE-position-state when sampling on device and do it exactly once per emitted token.
Make sure to account for different traces within a model.
When reducing sampled results across devices, compute the per-row stride from the gathered tensor's aligned page size, not the unpadded active-batch stride, or the reducer can select the wrong row.
To validate, run a focus overlap test under `--async-scheduling` and `sample_on_device_mode=all` checking tha the output passes the degenerate-output check, with no doubled subwords or repeated control tokens.

Leave prefix caching `False` unless it is implemented and tested.


If the vLLM plugin or harness is being changed, prefer the same safety rule there: overlap should default to false unless the model declares this proof-backed capability. Leaving overlap disabled may cost a few tokens/sec/user; letting it default on can silently corrupt generation.

The traced serving decode path reuses the full-model generator's canonical split-sampling path and replays via `ttnn.execute_trace(..., blocking=False)`. For `sample_on_device_mode=all`, serving has no new sampling strategy, host greedy/top-1 argmax, full-logits readback, generic top-k fallback for greedy, or Python readback/writeback token-feedback loop. If the full-model generator lacks split sampling, stop and fix `$full-model`; do not complete vLLM by patching sampling in the adapter. Do not copy a full page table every token when it is unchanged. Reduce token/current-position/page-table refresh to actual scheduler state changes, then prove both changed and unchanged cases with stale-input tests.

## Minimum-Surface Bring-Up Loop

Do not debug vLLM by repeatedly launching the complete all-layer model. First make the adapter work on the smallest representative serving target: the same generator, same `generator_vllm.py`, same plugin registration, same cache/page-table shapes, same terminal norm/LM head/sampling path, same trace behavior, and one real layer of each unique layer kind. Use that reduced target to make server launch, trace capture/replay, async split, vLLM cache ownership, on-device sampling, token/current-position/page-table refresh, and stale-input tests pass. Include a multi-request smoke test after the batch-1 path works, so batch/cache/page-table bugs are caught before the all-layer run.

The reduced target is only an inner-loop tool. It is not final serving evidence, and it must not be reported as the model's accuracy or performance. After the reduced target passes, run the complete all-layer model for the final accuracy, qualitative, sampling, and benchmark evidence.

Run checks from smallest to largest. If a vLLM pytest, prompt, request, or benchmark shape fails, rerun that failing item directly against a live server while debugging; do not rerun the whole suite after every edit. If a check suite is slow, run `--sampling-profile smoke` or a single targeted request first, then rerun the full profile once the issue is fixed.

## Plugin Registration

Register the model with the TT vLLM plugin. vLLM discovers TT models from the hardcoded list in:

```text
vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py::register_tt_models()
```

Add a `_register_model_if_missing(ModelRegistry, "TT<Arch>ForCausalLM", "<dotted.module.path>:<ClassName>")` call for the new adapter. Without this registration, the server will reject the architecture at startup before your adapter can run.

## vLLM Server Integration Test

Use the shared runner:

```bash
python -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/<model_name> \
  --hf-model <hf-model-id-or-local-path> \
  --mesh-device <N150|N300|T3K|TG> \
  --max-num-seqs <int> \
  --max-model-len <int> \
  --sampling-profile <full|smoke> \
  --tt-config '{"trace_region_size": <bytes>, "fabric_config": <fabric mode>}'
```

The runner owns server launch, health polling, check execution, and shutdown. It writes `server.log`, `sampling_tests.log`, `vllm_qualitative_outputs.json`, primary single-user raw `vllm_result.json`, primary normalized `vllm_benchmark.json`, `vllm_benchmark.log`, and by default the secondary CI serving-burst files `vllm_ci_serving_result.json`, `vllm_ci_serving_benchmark.json`, and `vllm_ci_serving_benchmark.log` under `<model_dir>/readiness_vllm/`.

`--stages` accepts `serve`, `sampling`, `qualitative`, and `benchmark`. The default runs the full launch-check-shutdown flow. To hold a server open while iterating:

```bash
python -m models.common.readiness_check.run_vllm_server \
  --stages serve \
  --model-dir ... --hf-model ... --mesh-device ... [--max-model-len ...] [--tt-config ...]
```

To attach checks to an existing server from another shell:

```bash
python -m models.common.readiness_check.run_vllm_server \
  --stages sampling \
  --max-num-seqs <int> \
  --sampling-profile <full|smoke> \
  --server-url http://localhost:8000 \
  --model-dir ... --hf-model ...
```

The runner enforces on-device sampling (`sample_on_device_mode: all` in the TT plugin config). The full-model stage must already provide traced token-out split sampling. If it does not, return to `$full-model`; this stage should only adapt that contract to vLLM.

If shared tests require host-side sampling, expose it as an explicit compatibility mode and label metrics from that mode separately. The optimized serving path still uses on-device traced sampling and `sample_on_device_mode=all`.

## No Tracy Or Perf-Report In vLLM Stages

Do not collect Tracy, `tt-perf-report`, or `TT_METAL_DEVICE_PROFILER` metrics from vLLM integration or optimized-vLLM serving runs. Do not set profiler env vars for the live server, do not run `python -m tracy` around `run_vllm_server`, do not run a serving-adapter profile just to produce a profiler table, and do not call `ttnn.ReadDeviceProfiler(mesh)` as part of vLLM-stage closure.

The vLLM stages have repeatedly wedged T3K machines during profiler/device-health closure. The useful vLLM evidence is the serving-path evidence produced by `run_vllm_server`: sampling results, qualitative outputs, degenerate-output checks, server logs, primary single-user `vllm bench serve` JSON with TTFT, TPOT, ITL, aggregate output throughput, and the secondary CI serving-burst `vllm bench serve` JSON for vLLM-nightly parity and serving-capacity context. Use existing full-model or reduced non-serving profiles from earlier stages for device-op/root-cause context. If those profiles are missing, record that evidence gap; do not recreate them inside the vLLM stage.

For optimized-vLLM, optimize with same-harness primary single-user and secondary CI serving-burst before/after metrics plus contract checks: async split, nonblocking trace replay, stale input coverage, on-device sampling, no host greedy argmax, no full-logits readback, and no unnecessary page-table/token/current-position refresh. Do not require Tracy, `tt-perf-report`, or live-serving device-profiler artifacts for vLLM-stage completion.

Check stages:

- `sampling`: runs the canonical TT plugin pytest suite against the live server. `--sampling-profile full` runs the whole suite; `--sampling-profile smoke` runs a small integration sanity subset for slow bring-up loops.
- `qualitative`: saves greedy and sampled completions for prompts from `models/common/readiness_check/vllm_prompts.txt`; use `$qualitative-check` to ensure the prompts are sent in the HF-declared model format, compare against controls, and judge coherence, topic, repetition, gibberish, and wrong-language drift.
- `benchmark`: runs the primary single-user decode profile by default: 128-token input, 128-token output, one prompt, `--max-concurrency 1`, `ignore_eos`, percentile metrics `ttft,tpot,itl,e2el`, and a completed-prompt gate. Use `vllm_benchmark.json` for headline decode t/s/u and comparisons to full-model or older agentic/custom-benchmark reports. This is the primary optimization target.
- `benchmark`: also runs the vLLM-nightly-shaped CI serving-burst profile by default: 100-token inputs, 100-token outputs, 32 prompts, no explicit `--max-concurrency`, `ignore_eos`, and the same metric set. Use `vllm_ci_serving_benchmark.json` for vLLM-nightly parity, serving-capacity context, and larger-batch/concurrency coverage. Do not use it as the headline decode t/s/u because burst admission and chunked prefill can affect TPOT. The readiness runner passes `--temperature 0.0` by default so both benchmark profiles are greedy. Use `--benchmark-use-server-generation-config` only when intentionally reproducing exact nightly/server-generation-config behavior, and label those numbers as sampled/default-generation-config rather than greedy single-user t/s/u.

When optimizing decode serving overhead, benchmark with the exact same runner, prompt/output lengths, `max_num_seqs`, model length, mesh, TT config, and sampling mode as the primary or previous comparison. Report raw vLLM `median_ttft_ms`/`p99_ttft_ms`, `mean_tpot_ms`/`p99_tpot_ms`, `median_itl_ms`/`p99_itl_ms`, `output_throughput`, and TPOT-derived decode t/s/u (`1000 / mean_tpot_ms`) before/after. Compare primary single-user 128/128/1 against primary single-user 128/128/1, and compare CI serving-burst 100/100/32 against CI serving-burst 100/100/32; do not treat those two workload shapes as direct perf verdicts against each other.

Keep teacher-forcing and serving performance separate. A readiness/PERF teacher-forcing number is useful as a decoder/generator lower bound; vLLM throughput includes serving orchestration, sampling, token feedback, request handling, and readback. If serving is much slower than teacher forcing, remove avoidable serving-specific overhead before retuning the decoder stack: fallback sampling, stale-input refreshes, per-token page-table copies, blocking trace replay, synchronizations, readbacks, and adapter-side reconstruction.

`--max-num-seqs` is passed to both server launch and sampling pytest (`--tt-max-num-seqs`). Do not leave it at 1 except for the primary single-user benchmark or a focused debugging run. Final serving evidence should include a larger value, normally up to 32, unless hardware or memory capacity prevents it.

For final vLLM-integration evidence, use `--sampling-profile full`. The normal debugging order is smoke first, then full: use `--sampling-profile smoke` for faster inner-loop iteration, rerun only failing pytest node ids or targeted requests while fixing failures, and run `--sampling-profile full` after the smoke and targeted checks pass. For MoE bring-up loops where the full profile is impractical, record the final status as `smoke-gated`; do not present it as equivalent to the full sampling gate unless the project owner explicitly accepts that coverage.

When determinism tests fail in vLLM, validate that logits output by the model for a given prompt are reproducible across runs and batch positions. Check both standalone model and running through vllm.

Classify reproducibility-only sampling failures separately when they are the only failures. Typical names include `test_top1_is_greedy`, `test_topk`, `test_uniform_seed_deterministic`, `test_specific_seed_reproducible`, `test_same_seeds_reproduce_across_batches`, `test_*_mixed_batch`, and `test_mixed_params_batch`. They do not require more serving-correctness work only when correctness, logprobs, crash-free serving, and qualitative output all pass. Correctness failures, missing logprobs, crashes, gibberish output, or wrong logprob values remain required work.

If vLLM crashes mid-run, kill leftover `EngineCore` or `vllm.entrypoints` processes before retrying; they can hold chip locks after `tt-smi -r`.

If a profiler run is accidentally started and fails, do not escalate it into repeated watcher/profiler/reset attempts. Kill leftover server processes. If `tt-smi -ls --local` or reset hangs, or if logs show remote Ethernet/ARC/ERISC failures such as `Timeout waiting for Ethernet core service remote IO request`, `ETH core heartbeat check failed`, `Unexpected ERISC Response Flags`, `Read 0xffffffff from ARC scratch`, or ARC lock/readback waits, stop profiler work, preserve the logs, mark the profiler evidence `hardware-profiler-limited`, and follow `$tt-device-usage` reset recovery before treating the stage as stopped. Request monitor/operator reboot only after bounded reset/list and mesh-smoke recovery fail or require authority this agent does not have.

Transient CCL/fabric link errors immediately after a failed multi-device run still need a device reset and one retry before being treated as hardware evidence, as long as `tt-smi` remains responsive and the failure is not part of the profiler/watcher pattern above.

Failed serving gates, missing serving artifacts, sampling failures, qualitative-output failures, and `$stage-review` `more-work-needed` findings are vLLM-stage work, not terminal stop reasons. If logs make the cause obvious, fix it directly and rerun the failing gate. If the first direct fix does not close the gate, or if the failure crosses the adapter, generator, cache ownership, scheduler inputs, trace inputs, sampling, or plugin registration path, use `$autofix`; it will run `$autodebug` if needed, then verify or refute each proposed bug before keeping any fix. Do not terminal-stop the vLLM stage until `$autofix` has tried and failed or an external dependency is genuinely unavailable.

Record the working server invocation in the work log, including `--max-model-len`, `--tt-config`, workload config, and any env vars that mattered. Use typed runner flags for `--max-model-len` and `--tt-config`; keep `--additional-server-args` for uncommon flags only.

## Output-Quality Verdicts Need A Control

Before classifying any serving-output problem as a model-quality limitation, use `$qualitative-check` to produce matching prompt-correct evidence from HF, or at minimum from the full-model stage on comparable prompts. If serving output is materially worse than the prompt-correct control, that is a serving regression and in scope for this stage - stale trace inputs, sampler state, and cache/position handling are the usual causes, not the checkpoint. If the shared readiness runner cannot send the correct prompt format, fix it or add a targeted prompt-correct request before judging output quality.

After qualitative collection, run:

```bash
python models/common/readiness_check/check_degenerate_output.py \
  --hf-model <hf-model-id> --missing-artifacts critical --scope vllm
```

Mechanical degeneracy - doubled tokens, single-token collapse - is never a model property. The runner-side stage gate runs the same check.

If the tokenizer has no chat template, say so explicitly, treat the checkpoint as a base model, and judge the qualitative outputs against continuation-style expectations; do not let chat-style prompts produce poor text that masks serving bugs.

## Evidence To Leave

Done means all of these are true and recorded:

- Full-model generator readiness baseline used before adding vLLM.
- Selected datatype policy loaded by serving, including KV-cache and CCL dtype.
- Adapter class, low-level generator methods it delegates to, and KV-cache ownership contract.
- Plugin registration path and architecture name.
- Exact successful `run_vllm_server` invocation.
- Served max context, matching `doc/context_contract.json`, with any hard-physical-limit reduction evidence.
- Non-aligned prompt-length evidence through serving: a valid request length that is not divisible by internal chunk/page/block alignment succeeds without capping or truncating the advertised context.
- Served batch/concurrency coverage, including the largest tested `max_num_seqs` up to 32 and any hard-physical-limit reduction evidence.
- Capability flags with evidence: no unproven `supports_async_decode=True`, no prefix-caching claim without tests, and on-device sampling verified for the measured mode.
- Evidence that serving uses the full-model split-sampling contract: internal sampling trace, `tt_out_tok` feedback into the persistent decode token input, greedy benchmarks using the fastest correct on-device sampling strategy measured for this mesh, and stale-token/current-position smoke coverage.
- Logit-determinism evidence through vLLM, with run-to-run and cross-batch-position reproducibility checks and standalone baseline comparison.
- Sampling test results, with any reproducibility-only failures separated from real failures.
- `$qualitative-check` artifacts: qualitative greedy and sampled serving-output verdict, prompt-format metadata, rendered prompts or token ids, and matching HF/full-model controls.
- Primary single-user benchmark workload config, temperature/generation-config mode, and whether it used default 128/128/1 shape or explicit overrides.
- Primary serving-path TTFT median/P99, TPOT mean/P99, ITL median/P99, aggregate output throughput, and TPOT-derived decode t/s/u from `vllm_benchmark.json`.
- Secondary CI serving-burst benchmark workload config and metrics from `vllm_ci_serving_benchmark.json` when comparing to vLLM nightly or serving-capacity evidence.
- Watcher or device-reset notes if relevant to serving stability.

## Preferred Outputs

Produce:

```text
models/autoports/<model>/tt/generator_vllm.py
models/autoports/<model>/doc/vllm_integration/work_log.md
models/autoports/<model>/doc/vllm_integration/README.md
```

The README should lead with vLLM sampling status, qualitative verdict, primary single-user 128/128/1 TTFT median, TPOT mean, ITL median, aggregate output throughput, and TPOT-derived decode t/s/u. Put CI serving-burst 100/100/32 metrics in a secondary section for vLLM-nightly parity and serving-capacity context. Include the workload config next to every number.

## Useful References

| Topic | Path |
|---|---|
| vLLM integration guide | `tech_reports/LLMs/vLLM_integration.md` |
| vLLM server integration runner | `models/common/readiness_check/run_vllm_server.py` |
| vLLM qualitative prompts | `models/common/readiness_check/vllm_prompts.txt` |
| Generator contract | `models/common/readiness_check/contract.py` |
| tt_transformers generator | `models/tt_transformers/tt/generator.py` |
| Thin vLLM adapter | `models/tt_transformers/tt/generator_vllm.py` |
| vLLM model registration | `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py::register_tt_models` |
