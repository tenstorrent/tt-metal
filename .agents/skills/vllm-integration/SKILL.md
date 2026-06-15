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

For decode performance, implement the vLLM async split before advertising it: `decode_forward(..., read_from_device=False)` should return device tensors, `read_decode_output(..., async_read=True)` should perform the minimal deferred read, and `process_decode_output_host(...)` should do host formatting. Only set `supports_async_decode=True` after this path passes the vLLM plugin's expectations with decode trace enabled and stale-token/current-position tests passing. Leave prefix caching `False` unless it is implemented and tested.

Do not treat `supports_async_decode=True` as permission for scheduler overlap. It only means the adapter can split submit/read/host-formatting. The steady overlap path is a separate contract: vLLM may build and submit decode step N+1 before sampled token N has been applied to host scheduler state. For a new adapter, set `tt_async_decode_allows_overlap = False` whenever next-step `model_input` is built from vLLM host state such as `input_batch.token_ids_cpu`, `num_tokens`, positions, or per-request scheduler tables, or whenever the adapter refreshes traced token/current-position/page-table tensors from that `model_input`.

Only set `tt_async_decode_allows_overlap = True` after a focused overlap test proves all of these under `--async-scheduling` and `sample_on_device_mode=all`:

- sampled token N is either applied to host scheduler state before step N+1 input construction, or step N+1 input construction is entirely device-owned and cannot be overwritten by stale host state;
- current-position/RoPE position state advances exactly once per emitted token and matches the request's output length;
- traced token/current-position/page-table inputs observed immediately before replay N+1 contain the new values, not the previous step's values;
- the async-overlap qualitative output passes the degenerate-output check, with no doubled subwords or repeated control tokens.

If the vLLM plugin or harness is being changed, prefer the same safety rule there: overlap should default to false unless the model declares this proof-backed capability. Leaving overlap disabled may cost a few tokens/sec/user; letting it default on can silently corrupt generation.

The traced serving decode path reuses the full-model generator's canonical split-sampling path and replays via `ttnn.execute_trace(..., blocking=False)`. For `sample_on_device_mode=all`, serving has no new sampling strategy, host greedy/top-1 argmax, full-logits readback, generic top-k fallback for greedy, or Python readback/writeback token-feedback loop. If the full-model generator lacks split sampling, stop and fix `$full-model`; do not complete vLLM by patching sampling in the adapter. Do not copy a full page table every token when it is unchanged. Reduce token/current-position/page-table refresh to actual scheduler state changes, then prove both changed and unchanged cases with stale-input tests.

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

The runner owns server launch, health polling, check execution, and shutdown. It writes `server.log`, `sampling_tests.log`, `vllm_qualitative_outputs.json`, and `vllm_benchmark.json` under `<model_dir>/readiness_vllm/`.

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

## No Tracy Or Perf-Report In vLLM Stages

Do not collect Tracy, `tt-perf-report`, or `TT_METAL_DEVICE_PROFILER` metrics from vLLM integration or optimized-vLLM serving runs. Do not set profiler env vars for the live server, do not run `python -m tracy` around `run_vllm_server`, do not run a serving-adapter profile just to produce a profiler table, and do not call `ttnn.ReadDeviceProfiler(mesh)` as part of vLLM-stage closure.

The vLLM stages have repeatedly wedged T3K machines during profiler/device-health closure. The useful vLLM evidence is the serving-path evidence produced by `run_vllm_server`: sampling results, qualitative outputs, degenerate-output checks, server logs, and benchmark JSON with TTFT, ITL, aggregate output throughput, and mean per-user decode t/s/u. Use existing full-model or reduced non-serving profiles from earlier stages for device-op/root-cause context. If those profiles are missing, record that limitation; do not recreate them inside the vLLM stage.

For optimized-vLLM, optimize with same-harness serving before/after metrics and contract checks: async split, nonblocking trace replay, stale input coverage, on-device sampling, no host greedy argmax, no full-logits readback, and no unnecessary page-table/token/current-position refresh. A vLLM stage is not incomplete merely because it lacks Tracy, `tt-perf-report`, or live-serving device-profiler artifacts.

Check stages:

- `sampling`: runs the canonical TT plugin pytest suite against the live server. `--sampling-profile full` runs the whole suite; `--sampling-profile smoke` runs a small integration sanity subset for slow bring-up loops.
- `qualitative`: saves greedy and sampled completions for prompts from `models/common/readiness_check/vllm_prompts.txt`; read the outputs and judge coherence, topic, repetition, gibberish, and wrong-language drift. See `Output-Quality Verdicts Need A Control` below before classifying any problem as model-intrinsic.
- `benchmark`: runs the configured synthetic workload and records TTFT P50/P99, ITL P50/P99, aggregate output throughput, and mean per-user decode t/s/u.

When optimizing decode serving overhead, benchmark with the exact same runner, prompt/output lengths, `max_num_seqs`, model length, mesh, TT config, and sampling mode as the canonical or previous comparison. Report TTFT, ITL, output throughput, and mean per-user decode t/s/u before/after; compare directly to the canonical same-machine implementation when it is available.

Keep teacher-forcing and serving performance separate. A readiness/PERF teacher-forcing number is useful as a decoder/generator lower bound; vLLM throughput includes serving orchestration, sampling, token feedback, request handling, and readback. If serving is much slower than teacher forcing, remove avoidable serving-specific overhead before retuning the decoder stack: fallback sampling, stale-input refreshes, per-token page-table copies, blocking trace replay, synchronizations, readbacks, and adapter-side reconstruction.

`--max-num-seqs` is passed to both server launch and sampling pytest (`--tt-max-num-seqs`).

For final vLLM-integration evidence, use `--sampling-profile full`. Use `--sampling-profile smoke` for faster inner-loop iteration. For batch-1 MoE bring-up loops, `--sampling-profile smoke` is acceptable as the final sampling gate and `full` may be skipped entirely because it is very slow in that regime.

When determinism tests fail in vLLM, validate that logits output by the model for a given prompt are reproducible across runs and batch positions. Check both standalone model and running through vllm.

Reproducibility-only sampling failures are out of scope when they are the only failures. Typical names include `test_top1_is_greedy`, `test_topk`, `test_uniform_seed_deterministic`, `test_specific_seed_reproducible`, `test_same_seeds_reproduce_across_batches`, `test_*_mixed_batch`, and `test_mixed_params_batch`. Correctness failures, missing logprobs, crashes, gibberish output, or wrong logprob values remain in scope.

If vLLM crashes mid-run, kill leftover `EngineCore` or `vllm.entrypoints` processes before retrying; they can hold chip locks after `tt-smi -r`.

If a profiler run is accidentally started and fails, do not escalate it into repeated watcher/profiler/reset attempts. Kill leftover server processes. If `tt-smi -ls --local` or reset hangs, or if logs show remote Ethernet/ARC/ERISC failures such as `Timeout waiting for Ethernet core service remote IO request`, `ETH core heartbeat check failed`, `Unexpected ERISC Response Flags`, `Read 0xffffffff from ARC scratch`, or ARC lock/readback waits, stop the stage, preserve the logs, mark the result `hardware-profiler-limited`, and ask the monitor to reboot the physical Docker host. On `wh-lb-90`, direct `sudo /sbin/reboot` on the host restored all 8 devices and full 2x4 mesh open; `ird reboot --force` had reported success but did not reset host uptime or recover `tt-smi`.

Transient CCL/fabric link errors immediately after a failed multi-device run still need a device reset and one retry before being treated as hardware evidence, as long as `tt-smi` remains responsive and the failure is not part of the profiler/watcher pattern above.

If serving behavior fails in a way that crosses the adapter, generator, cache ownership, scheduler inputs, or plugin registration path and ordinary log reading does not explain it, use `$autofix` before turning the vLLM stage into broad full-model debugging; it will run `$autodebug` if needed, then verify or refute each proposed bug before keeping any fix.

Record the working server invocation in the work log, including `--max-model-len`, `--tt-config`, workload config, and any env vars that mattered. Use typed runner flags for `--max-model-len` and `--tt-config`; keep `--additional-server-args` for uncommon flags only.

## Output-Quality Verdicts Need A Control

Before classifying any serving-output problem as a model-quality limitation, produce matching evidence: the same prompts through the HF reference, or at minimum the full-model stage's free-running generation on comparable prompts. If serving output is materially worse than what the model produced at the full-model stage, that is a serving regression and in scope for this stage - stale trace inputs, sampler state, and cache/position handling are the usual causes, not the checkpoint.

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
- Capability flags with evidence: no unproven `supports_async_decode=True`, explicit `tt_async_decode_allows_overlap` value with proof if true, no prefix-caching claim without tests, and on-device sampling verified for the measured mode.
- Evidence that serving uses the full-model split-sampling contract: internal sampling trace, `tt_out_tok` feedback into the persistent decode token input, greedy benchmarks using the fastest correct on-device sampling strategy measured for this mesh, and stale-token/current-position smoke coverage.
- Logit-determinism evidence through vLLM, with run-to-run and cross-batch-position reproducibility checks and standalone baseline comparison.
- Sampling test results, with any reproducibility-only failures separated from real failures.
- Qualitative greedy and sampled serving-output verdict.
- Serving benchmark workload config.
- Serving-path TTFT P50/P99, ITL P50/P99, aggregate output throughput, and mean per-user decode t/s/u.
- Watcher or device-reset notes if relevant to serving stability.

## Preferred Outputs

Produce:

```text
models/autoports/<model>/tt/generator_vllm.py
models/autoports/<model>/doc/vllm_integration/work_log.md
models/autoports/<model>/doc/vllm_integration/README.md
```

The README should lead with vLLM sampling status, qualitative verdict, TTFT P50, ITL P50, and mean per-user decode t/s/u, including the workload config used.

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
