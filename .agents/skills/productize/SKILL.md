---
name: productize
description: Take an existing ported TTNN decoder under models/autoports/<model> and build the full model wrapper, generator contract, vLLM adapter, and readiness-check path needed to drive the model end-to-end.
user_invocable: true
---

# Productize

## Mission Context

Read `.agents/notes/model-bringup-mission.md` first. This stage starts from a working decoder and turns it into a model that can be driven end-to-end on real weights! The important outcome is a full, working model with a generator and vLLM integration.

## Your Part

Starting from `models/autoports/<model>/`, add:

```text
tt/model.py
tt/generator.py
tt/generator_vllm.py
```

Use the single decoder implementation from the latest stage as your starting point e.g. multi-chip if it exists, otherwise optimized, otherwise functional. Use the same mesh device as the decoder implementation for your work.

## When porting a new model

1. Read and understand the source decoder + supporting modules under `models/autoports/<model_name>/`.
2. `<model_dir>/tt/generator.py` implementing the `models.common.readiness_check.contract.Generator` ABC.
3. `<model_dir>/tt/generator_vllm.py` — thin delegate for tt-transformers-family models; thick (custom prefill/decode) for DeepSeek-style. See the SKILL.md for the vLLM plugin contract.
4. Reference file generated via `python -m models.common.readiness_check.generate ...`, saved under `models/common/readiness_check/references/<model_name>.refpt`.
5. Verify via the three readiness checks: `run_prefill_check` and `run_teacher_forcing` (both scored numerically as top-1 / top-5 / top-100 hit rates against the HF reference), plus `run_autoregressive` (free-running side-by-side completion against the HF reference — no programmatic check; you read both completions and judge whether the ported model's output looks reasonable).

The shared readiness check expects **single-prompt, greedy/argmax** semantics for deterministic top-K hit-rate comparison against the HF teacher.

## Inputs To Check

| Input | Where to find it |
|---|---|
| Model directory | `models/autoports/<model_name>/` containing the ported decoder block and supporting modules. |
| HF model id or local checkpoint | Usually in `config.py` as `HF_MODEL_ID`; prefer local checkpoint paths under `/proj_sw/user_dev/` when available. |
| Mesh or device label | Usually in `config.py` as `MESH_DEVICE`; use the same mesh shape as the decoder implementation. |
| Sampling expectations | The readiness path uses greedy/argmax behavior; vLLM may later exercise stochastic sampling. |

When running readiness checks, choose `--mesh-device` from the model's configured target and the hardware actually available. If the model target is available, use it. If not, use the largest available compatible mesh and record the mismatch.

If the model architecture is too recent for the current `transformers` install, update the local environment enough to load the HF config/reference and record that in the work log.

## Model Wrapper

Write a `tt/model.py` that creates a decoder stack and produces a full implementation of the huggingface reference model using only TTNN code. Read the HuggingFace reference model carefully as this will involve porting extra operations to TTNN, for example:

- embedding;
- final norm;
- LM head, respecting tied embeddings when the HF config uses them;
- paged KV-cache exposure;
- on-device sampling (reuse models.common.sampling if at all possible here)
- logit return path expected by the generator.

Avoid TTNN weight caching (from_torch is now fast enough).

## Generator

`tt/generator.py` is the load-bearing file. It should expose `build_generator(model_dir, mesh_device, **kwargs)` and a concrete class that subclasses `models.common.readiness_check.contract.Generator` (the ABC).

Implement both API levels:

- low-level `prefill_forward(...)` and `decode_forward(...)` with caller-managed KV cache and page table;
- high-level `generate(prompt_token_ids, max_new_tokens, *, next_input=None, **kwargs)` with generator-managed KV cache.

The high-level path should be a thin deterministic loop over the low-level methods. Readiness uses greedy/argmax behavior for top-K comparison against the HF teacher. On-device sampling is preferred when the model path supports it, but correctness of the readiness path is the priority.

The generator should own tokenizer loading, page-table setup, KV-cache reset, and any trace-side state reset needed between runs. Keep vLLM-facing low-level methods explicit enough that `generator_vllm.py` can delegate to them rather than duplicating model logic.

Decode tracing should be maintained as a fully-traced decode path that does not need host involvement within a forward pass. If necessary you may do a nominal amount of host work between steps e.g. copying the output tensor to the input, incrementing the position, but it should not be necessary to move tensors back to the host for this and where possible they should be part of the trace. If in doubt check what models/tt_transformers does.

## vLLM Adapter

`tt/generator_vllm.py` should be a thin adapter when the generator's low-level methods already match vLLM needs, and a thicker adapter only when the model shape contract requires it.

Read:

- `tech_reports/LLMs/vLLM_integration.md`;
- `models/tt_transformers/tt/generator_vllm.py`;

Add a local registration hook in `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`.

## Verification

Use the shared model-readiness check:

```bash
python -m models.common.readiness_check.generate ...
python -m models.common.readiness_check.run_prefill_check ...
python -m models.common.readiness_check.run_teacher_forcing ...
python -m models.common.readiness_check.run_autoregressive ...
```

You will need to provide the readiness check with the correct mesh device shape to match the decoder and your model.

Reuse an existing reference file when it matches the prompt set and token lengths. Generate a new one when it does not. Report top-1, top-5, and top-100 hit rates for prefill and teacher-forcing decode. Differences between prefill and decode are debugging signal: cache, position, page table, or sampling behavior often explains them. We expect top-5 ≥ 98%, top-100 = 100%. Lower top-1 (~90%) is expected from bf8 quantization. If we're getting lower than these figures then first investigate your own new model code for bugs and if that is clean dive into the decoder and try increasing the datatype precision / compute fidelity to see if that improves accuracy. It's our job to deliver a fully working model, so if the decoder we have been given needs to be fixed then we're the ones who will fix it!

If no reference exists yet, generate one:

```bash
python -m models.common.readiness_check.generate \
  --hf-model <hf-model-id-or-local-path> \
  --prompt-len 128 \
  --gen-len 256 \
  --output models/common/readiness_check/references/<model>.refpt
```

Run the complementary readiness checks:

```bash
python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/<model_name> \
  --reference models/common/readiness_check/references/<model>.refpt \
  --mesh-device <N150|N300|T3K|TG>

python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/<model_name> \
  --reference models/common/readiness_check/references/<model>.refpt \
  --mesh-device <N150|N300|T3K|TG>

python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/<model_name> \
  --hf-model <hf-model-id-or-local-path> \
  --mesh-device <N150|N300|T3K|TG>
```

The first two are scored numerically (top-1 / top-5 / top-100). The third — `run_autoregressive` — has both the HF reference and the ported model generate a completion to the same prompt (loaded from `models/common/readiness_check/autoregressive_prompt.txt`) and writes `hf_completion.txt` and `tt_completion.txt` side by side under `<model_dir>/readiness_autoregressive/`. There is **no programmatic check** — you must read both completions yourself and judge whether the ported model's output is reasonable given the reference. Expect minor lexical drift from bf8 quantization; you're looking for coherent, on-topic continuation, not token-exact match. Severe divergence (incoherent text, immediate repetition, wrong language, runs of identical tokens) means the model is broken even if teacher-forcing top-100 looks healthy. Include the verdict — and a short excerpt from each — in the work log.

You must also run and generate the following performance figures from warmed-up sessions with sampling on-device enabled:
1. 128-token prefill time-to-first-token (ms)
2. End-to-end decode t/s/u (average over next 32 tokens following above prefill, as observed at the host)

Before treating productize as done, do a run with watcher enabled and record the status. Use `TT_METAL_WATCHER=10`. You can use `run_autoregressive`.

### vLLM Server Integration Test

After the three readiness checks above pass, verify the vLLM serving path.

**Prerequisite — register the model with the plugin.** vLLM's TT plugin discovers models from a hardcoded list in `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py::register_tt_models()`. A new `<model_dir>/tt/generator_vllm.py` is **not** auto-discovered — you must add a `_register_model_if_missing(ModelRegistry, "TT<Arch>ForCausalLM", "<dotted.module.path>:<ClassName>")` call there. Without that line, the server will refuse the architecture at startup with "architecture not in TT registry" and the runner cannot exercise your adapter.

Once registered:

```bash
python -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/<model_name> \
  --hf-model <hf-model-id-or-local-path> \
  --mesh-device <N150|N300|T3K|TG> \
  --max-model-len <int>                            # if the model caps below vLLM's default \
  --tt-config '{"trace_region_size": <bytes>, ...}'  # if the model needs specific TT plugin tuning
```

The runner **enforces on-device sampling** (`sample_on_device_mode: all` in the TT plugin config). A ported model that cannot serve sampling from the device is not production-ready — making the model wrapper expose its on-device sampler is part of this stage, not a follow-up. Your `tt/model.py` must wire `models.common.sampling` (or an equivalent on-device sampler) into the decode path; the readiness runner will fail fast if it can't. `--tt-config` merges into the runner defaults, so callers can extend (e.g. add `trace_region_size`, `fabric_config`) but `sample_on_device_mode` is always set.

The runner owns the launch — env vars, CLI flags, server-ready polling, fast-fail markers, and shutdown all live inside it. You should not need to read the workflow files to invoke it. Once `/health` is up (default budget: 20 minutes; trace compile can take 10+ on first start), it runs:

1. **Sampling tests** — invokes `pytest vllm/plugins/vllm-tt-plugin/tests/tt/` against the running server. These are the canonical plugin tests (greedy determinism, seeded reproducibility, seed variety, logprobs, penalties, request isolation). Pass/fail is programmatic; on failure the runner stops and dumps the server-log tail.
2. **Qualitative completions** — runs prompts from `models/common/readiness_check/vllm_prompts.txt` with both greedy (`temperature=0`) and sampled (`temperature=0.7, top_p=0.9`) settings, saving them to `<model_dir>/readiness_vllm/vllm_qualitative_outputs.json` for **manual review**. Read each prompt and both completions and check:
   - both outputs coherent and on-topic
   - no repetition loops, gibberish, or wrong-language drift
   - greedy and sampled both reasonable

Include your verdict in the work log. The runner writes `server.log`, `sampling_tests.log`, and `vllm_qualitative_outputs.json` under `<model_dir>/readiness_vllm/`; debug failures from those. If vLLM crashed mid-run, kill any leftover `EngineCore` / `vllm.entrypoints` zombie processes before retrying — they hold chip locks even after `tt-smi -r`.

**Reproducibility-only failures are out of scope.** There are known framework bugs that cause sampled outputs to not be bit-exact across runs, positions, or batches. Tests that *only* assert exact reproducibility — typical names: `test_top1_is_greedy`, `test_topk`, `test_uniform_seed_deterministic`, `test_specific_seed_reproducible`, `test_same_seeds_reproduce_across_batches`, `test_*_mixed_batch`, `test_mixed_params_batch` — can fail for this reason on any model and are not your problem to fix. If those are the **only** failures, note them in the work log and move on. Failures involving actual correctness (gibberish output, wrong logprobs *values*, missing logprobs entirely, crashes) are still in scope.

**Record the working server invocation in the work log.** Once you have the runner passing end-to-end, write down the exact `--max-model-len` and `--tt-config` values you used (e.g. `--max-model-len 32768 --tt-config '{"trace_region_size": 0, "fabric_config": "FABRIC_1D"}'`). Future runs, handoffs, and CI submissions all need this — discovering it by trial-and-error each time is wasted hardware time. If you also discovered env vars that mattered (e.g. `TT_LLAMA_TEXT_VER`), record those too.

`--max-model-len` and `--tt-config` are typed first-class flags on the runner — use those rather than smuggling `--max_model_len` or `--plugin-config` through `--additional-server-args`. Reserve `--additional-server-args` for genuinely uncommon flags (e.g. `--async-scheduling`, `--tokenizer <path>`).

## Two-level Generator API

Every per-model generator exposes **both** API levels in one file:

- **Low-level** — `prefill_forward(tokens, *, page_table, kv_cache, prompt_lens, ...)` and `decode_forward(tokens, start_pos, *, page_table, kv_cache, ...)`. Caller manages KV cache and page table. Used by `generator_vllm.py` (vLLM owns KV cache).
- **High-level** — `generate(prompt_token_ids, max_new_tokens, *, next_input=None)`. Generator owns KV cache and page table internally; caller sees only token IDs in/out. Used by demos and the readiness check.

The high-level method is implemented as a thin loop over the low-level methods. No separate "adapter" file; same file serves both.

## Factory by convention

Each per-model directory exposes a module-level `build_generator(model_dir, mesh_device, **kwargs)` function in `<model_dir>/tt/generator.py`. Runners discover it by convention (`importlib.util.spec_from_file_location`) — no manifest file, no plugin registry.

`model_dir` is kept in the signature even when currently unused, so future per-model config files can be wired in without an interface change.

## Preferred Outputs

We vary slightly from the standard outputs defined in `.agents/notes/output-files.md` for this final stage. Produce:

```text
models/autoports/<model>/tt/model.py
models/autoports/<model>/tt/generator.py
models/autoports/<model>/tt/generator_vllm.py
models/autoports/<model>/doc/productize/work_log.md
models/autoports/<model>/doc/productize/README.md
```

Your README should lead with the top-1/top-5/top-100 figures for the full model and the 128-token prefill ms and mean over 32 token decode t/s/u for the full model.

## Useful References

| Topic | Path |
|---|---|
| Generator contract | `models/common/readiness_check/contract.py` |
| Readiness reference generator | `models/common/readiness_check/generate.py` |
| Batch prefill readiness runner | `models/common/readiness_check/run_prefill_check.py` |
| Teacher-forcing readiness runner | `models/common/readiness_check/run_teacher_forcing.py` |
| Autoregressive side-by-side runner | `models/common/readiness_check/run_autoregressive.py` |
| vLLM server integration runner | `models/common/readiness_check/run_vllm_server.py` |
| Autoregressive prompt | `models/common/readiness_check/autoregressive_prompt.txt` |
| vLLM qualitative prompts | `models/common/readiness_check/vllm_prompts.txt` |
| tt_transformers model wrapper | `models/tt_transformers/tt/model.py::Transformer` |
| Paged KV allocation pattern | `models/tt_transformers/tt/attention.py::init_kv_cache` |
| tt_transformers generator | `models/tt_transformers/tt/generator.py` |
| Thin vLLM adapter | `models/tt_transformers/tt/generator_vllm.py` |
| vLLM model registration | `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py::register_tt_models` |
