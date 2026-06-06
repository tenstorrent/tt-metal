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

## vLLM Adapter

`tt/generator_vllm.py` should delegate to the existing generator's low-level methods wherever possible. Keep it thin unless the model shape contract genuinely requires a thicker adapter.

Read:

- `tech_reports/LLMs/vLLM_integration.md`;
- `models/tt_transformers/tt/generator_vllm.py`;
- the model's existing `tt/generator.py` low-level `prefill_forward` and `decode_forward` methods.

vLLM owns KV-cache allocation in serving mode. Preserve the generator's two cache ownership modes:

- standalone readiness/generator mode, where the generator owns cache allocation and reset;
- vLLM mode, where `allocate_vllm_kv_cache` creates the cache and the adapter passes that exact cache through the low-level API.

Make prompt lengths, page tables, decode positions, batch dimensions, trace-side state, and on-device sampling explicit. The serving decode pass must drive the generator's traced decode path, not an eager-only fallback. When adding or debugging trace capture/replay, trace-safe inputs, or replay correctness for this adapter, use `$tt-enable-tracing`. The adapter should not duplicate model logic that already lives in `tt/model.py` or `tt/generator.py`.

For decode performance, implement the vLLM async split before advertising it: `decode_forward(..., read_from_device=False)` should return device tensors, `read_decode_output(..., async_read=True)` should perform the minimal deferred read, and `process_decode_output_host(...)` should do host formatting. Only set `supports_async_decode=True` after this path passes the vLLM plugin's expectations with decode trace enabled. Leave prefix caching `False` unless it is implemented and tested.

The traced serving decode path should reuse persistent trace inputs and replay via `ttnn.execute_trace(..., blocking=False)`. For `sample_on_device_mode=all`, keep sampling on device using the common sampling trace support; if the model trace returns sampler-ready logits, pass them as already prepared rather than rebuilding or reading full logits on the host. Remove host greedy/top-1 argmax fast paths, or prove they are not used by the measured benchmark. Avoid copying a full page table every token when it is unchanged, but add a stale-input test before reducing any token/position/page-table refresh.

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

The runner enforces on-device sampling (`sample_on_device_mode: all` in the TT plugin config). If the full-model stage left sampler work incomplete, finish the smallest necessary model/generator change here and record it.

Check stages:

- `sampling`: runs the canonical TT plugin pytest suite against the live server. `--sampling-profile full` runs the whole suite; `--sampling-profile smoke` runs a small integration sanity subset for slow bring-up loops.
- `qualitative`: saves greedy and sampled completions for prompts from `models/common/readiness_check/vllm_prompts.txt`; read the outputs and judge coherence, topic, repetition, gibberish, and wrong-language drift.
- `benchmark`: runs the configured synthetic workload and records TTFT P50/P99, ITL P50/P99, aggregate output throughput, and mean per-user decode t/s/u.

When optimizing decode serving overhead, benchmark with the exact same runner, prompt/output lengths, `max_num_seqs`, model length, mesh, TT config, and sampling mode as the canonical or previous comparison. Report TTFT, ITL, output throughput, and mean per-user decode t/s/u before/after; compare directly to the canonical same-machine implementation when it is available.

`--max-num-seqs` is passed to both server launch and sampling pytest (`--tt-max-num-seqs`).

For final vLLM-integration evidence, use `--sampling-profile full`. Use `--sampling-profile smoke` for faster inner-loop iteration. For batch-1 MoE bring-up loops, `--sampling-profile smoke` is acceptable as the final sampling gate and `full` may be skipped entirely because it is very slow in that regime.

Reproducibility-only sampling failures are out of scope when they are the only failures. Typical names include `test_top1_is_greedy`, `test_topk`, `test_uniform_seed_deterministic`, `test_specific_seed_reproducible`, `test_same_seeds_reproduce_across_batches`, `test_*_mixed_batch`, and `test_mixed_params_batch`. Correctness failures, missing logprobs, crashes, gibberish output, or wrong logprob values remain in scope.

If vLLM crashes mid-run, kill leftover `EngineCore` or `vllm.entrypoints` processes before retrying; they can hold chip locks after `tt-smi -r`.

If serving behavior fails in a way that crosses the adapter, generator, cache ownership, scheduler inputs, or plugin registration path and ordinary log reading does not explain it, use `$autofix` before turning the vLLM stage into broad full-model debugging; it will run `$autodebug` if needed, then verify or refute each proposed bug before keeping any fix.

Record the working server invocation in the work log, including `--max-model-len`, `--tt-config`, workload config, and any env vars that mattered. Use typed runner flags for `--max-model-len` and `--tt-config`; keep `--additional-server-args` for uncommon flags only.

## Evidence To Leave

Final vLLM integration evidence should show:

- Full-model generator readiness baseline used before adding vLLM.
- Adapter class, low-level generator methods it delegates to, and KV-cache ownership contract.
- Plugin registration path and architecture name.
- Exact successful `run_vllm_server` invocation.
- Capability flags with evidence: no unproven `supports_async_decode=True`, no prefix-caching claim without tests, and on-device sampling verified for the measured mode.
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
