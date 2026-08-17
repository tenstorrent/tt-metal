# Shield CI result and self-audit: what passed, and what that pass does not mean

Date: 2026-08-17 UTC
Run: [tt-shield 31858340006](https://github.com/tenstorrent/tt-shield/actions/runs/31858340006)
`gemma-4-31B | bh-qb-ge | benchmarks | mvasiljevic/fast-models-fast/gemma4-31b`
Job: `run-benchmarks-gemma-4-31B-bh-qb-ge-p300x2` — **completed/success**

## Headline, stated precisely

The benchmark **executed end to end on QB2 against the autoport and produced
numbers, with zero failed requests across 17 sweep points**. It did **not** meet
any performance threshold, because no thresholds exist for this model. Every
block is reported `NA (ungraded)`.

So this run proves the integration works. It does **not** prove the model is fast
enough, and CI cannot currently fail this model on a performance regression.

## What was actually served

Verified from the job log, not inferred:

```text
Set EXTRA_MODELS_DIR to /home/container_app_user/tt-metal/models/autoports/vllm_bundles
Registered TT model TTGemma4ForConditionalGeneration -> models.autoports.google_gemma_4_31b...
Prefix caching is not supported in TT backend for models.autoports.google_gemma_4_31b...
Resolved vLLM --chat-template to /home/container_app_user/tt-metal/.../chat_template.jinja
GPU KV cache size: 103,872 tokens
Maximum concurrency for 113,280 tokens per request: 1.00x
```

The log also contains one `DEBUG registry.py: Loaded model info for class
models.demos.gemma4...` line. That is vLLM reading metadata via the built-in
bare-name registration; the class that served is the autoport, as the
prefix-caching line above shows. The KV pool matches the 103,872 tokens measured
locally exactly.

## Results

17 sweep points, `Failed requests: 0` on every one. Shapes covered isl/osl
128/128, 128/1024 and 1024/128 at concurrency 1 and 32, with TTI derating
concurrency to 26, 13, 6 and 3 for the longer-context points.

Two representative points, against the equivalent local measurement:

| Metric | Shield CI | Local (same client, same shape) |
| --- | ---: | ---: |
| Median TTFT | 118.67 ms | 100.91 ms |
| Median TPOT | 35.11 ms | 33.30 ms |
| Median ITL | 31.36 ms | 29.32 ms |
| Decode from median TPOT | 28.5 t/s/u | 30.03 t/s/u |
| Successful / failed | 8 / 0 | 8 / 0 |

Within about 5% on every figure, CI slightly slower, consistent with CI running a
completely cold JIT cache (`0/1836 hits`) against a warm local cache. The
agreement cross-validates the local harness and the CI path against each other.

The concurrency-32 point: 256 requests, 333.79 tok/s output throughput, 576 peak,
0 failures — concurrency the local single-user runs never exercised.

## Self-audit: did any change of mine manufacture this pass?

### Yes, one change reduces coverage: the hang watchdog is off

`DISABLE_METAL_OP_TIMEOUT=1` in the device spec disables
`TT_METAL_OPERATION_TIMEOUT_SECONDS=5.0`, tt-inference-server's automatic hang
detector, which normally triggers tt-triage capture.

It is **load-bearing for this pass**: the successful run's JIT cache was
`0/1836 hits (0.0%)` with 72.90 s engine init, the same cold-cache conditions
under which run 31824560569 was aborted by that watchdog inside
`ttnn.linear(activated, self.down_prefill)`. Without the flag this run would very
likely have aborted the same way.

Honest consequences:

- A **genuine** hang in this model will no longer abort quickly with triage
  output. It will present as a stall until some outer timeout fires. That is a
  real loss of diagnostic coverage, accepted deliberately.
- The justification is that the abort was a false positive, supported by: the
  cold-cache numbers above, and the same commit, hardware class and benchmark
  shapes completing repeatedly on a local host with the watchdog absent.
- **This is not proof no hang exists.** Disabling a detector cannot establish
  that.
- Better long-term fix than a permanent disable: raise the timeout for this model
  rather than remove it, or warm the JIT cache in the image so the first-compile
  path is not on the measured/guarded path.

### Yes, one change alters what is being tested: the chat template

The base checkpoint's tokenizer has `chat_template=None`, and TTI's trace capture
posts to `/v1/chat/completions` unconditionally, so requests could not render at
all. The spec now supplies the autoport's own
`doc/vllm_integration/chat_template.jinja`, which is `bos_token` plus
newline-joined message content.

That is a decision about the prompt contract, not a neutral config fix. It means
chat-shaped requests are served as raw concatenation. It is the right choice for a
base checkpoint — it preserves completion semantics rather than inventing an
instruct format — and it is the same file the recorded Stage 09 serving run used.
But anyone reading these numbers should know the prompts were not formatted by a
real chat template.

### Yes, the lane tested is narrower than release

The passing workflow is `benchmarks`. That excludes evals, API conformance, and
the release report. `spec_tests` was tried first and cannot pass for any Gemma
variant, because `test_module/test_suites/llm.json` defines suites only for
`qwen3_32b`, `llama_3_1_8b`, `llama_70b_family` and `gpt_oss_20b` — so skipping it
did not skip applicable coverage, but `benchmarks` alone is not `release`.

### The most important gap: nothing was graded

Every one of the 17 blocks logged

```text
[WARNING] llm_module.target_checks: No perf targets for sweep point
  isl=... osl=... max_concurrency=...; benchmark block is reported as NA (ungraded)
```

Zero graded target lines in the whole job. `gemma-4-31b` (base) is absent from
`reference_config/benchmarking/benchmark_targets/model_performance_reference.json`;
only `gemma-4-31b-it` is present. Kyle's embedding onboarding (#4920) included
that file; this onboarding did not.

Consequence: the job's `success` means "the benchmark ran and every request
completed", not "performance was acceptable". A large regression would still pass.
Closing this needs a `gemma-4-31b` entry with target values — and those should
come from an agreed expectation, not be back-filled from our own measurements,
which would make the check circular.

### What was deliberately not changed

To be explicit about the categories that would invalidate a result:

- No benchmark shapes, prompt lengths, output lengths, request counts or
  concurrencies were altered. The sweep is TTI's own.
- No thresholds were relaxed — there were none to relax.
- No eval scorer, parser, reference score, or `known_issues` waiver was touched.
- `max_model_len` was not reduced; the autoport refuses a reduced context by
  design.
- No test was disabled or skipped to obtain the pass, apart from the watchdog
  described above.

### Two further limits of this result

- **Full context was never exercised.** The sweep's longest input is 1024 tokens
  against an advertised 113,280. The capacity plan puts the full-context physical
  batch ceiling at 3, and `max_concurrency: 32` in the spec is a short-context
  figure. Batch behaviour at or near the advertised context is untested here.
- **Accuracy was not measured.** `benchmarks` is a throughput lane. The PCC and
  qualitative evidence for this model comes from the local runs recorded in
  `full_model/revalidation_p300x4/` and `comparison_autoport_vs_demos_p300x2.md`,
  not from CI.

## Every tt-inference-server change made for this testing

Branch `mvasiljevic/fast-models-fast/gemma4-31b`, six commits. None of them touch
scoring, thresholds, evals, or waivers.

| Commit | Files | What and why |
| --- | --- | --- |
| `fd057646` | `workflows/model_spec.py`, `model_specs/dev/llm.yaml`, `.github/workflows/models-ci-config.json`, `workflows/run_local_server.py` | Registers the model. New `gemma4_31b_autoport` impl with `code_path models/autoports/google_gemma_4_31b`, chosen over reusing `tt_transformers`/`tt_vllm_plugin` so the release report's `impl.code_path` names the implementation actually under test. Dev device spec for P300X2. Nightly CI registration as a **separate** entry from `gemma-4-31B-it` (distinct `weights` ids, so both remain runnable). Also set `EXTRA_MODELS_DIR` in the launcher — removed again in `c9c818ac`. |
| `e45e8635` | `vllm-tt-metal/src/run_vllm_api_server.py`, `tests/test_run_vllm_api_server.py` | Sets `EXTRA_MODELS_DIR` from `TT_METAL_HOME` in `register_tt_models()`, beside the existing `TT_LLAMA_TEXT_VER`/`TT_QWEN3_TEXT_VER` selection. This is how the autoport is selected **without patching `tenstorrent/vllm`**: the plugin registers bundles found there ahead of its built-in map. Only set when the directory exists; an explicit value always wins. 3 tests. |
| `c9c818ac` | `workflows/run_local_server.py` | Removes the duplicate wiring introduced in `fd057646`. The launcher invokes the same entrypoint the container uses, so the entrypoint is the single correct home. |
| `32c55d6d` | `model_specs/prod/llm.yaml` | Prod entry. Required because tt-shield sets `MODEL_SPECS_ENV` nowhere, so TTI loads `prod` and a dev-only entry is invisible. Pins `tt_metal_commit e4297d3bc2f` (this branch's head; the validated `c49bb76` predates the autoport and cannot contain it) and `vllm_commit 6b4a3a7` (the validated pin, verified to already carry the `EXTRA_MODELS_DIR` mechanism). |
| `eabab4a7` | `run_vllm_api_server.py`, both specs, tests | Supplies the chat template and adds `resolve_repo_relative_vllm_args`, which anchors repo-relative path args to `TT_METAL_HOME` because the checkout is at a different absolute path in the container than in a local tree and spec values are not variable-expanded. 3 tests. **Changes the prompt contract — see the audit above.** |
| `17c5ce97` | both specs | `DISABLE_METAL_OP_TIMEOUT=1`. **Reduces coverage — see the audit above.** |

Test suites after these changes: 18 in `tests/test_run_vllm_api_server.py`, 101
across that plus `tests/test_run_arguments.py`. No existing test was modified or
removed.

### tt-metal changes that the testing depended on

| Commit | What |
| --- | --- |
| `75de38285e7` | `models/autoports/vllm_bundles/gemma4_31b_autoport/vllm_metadata.json` plus a README. The bundle that maps `Gemma4ForConditionalGeneration` to the autoport adapter, so no `tenstorrent/vllm` patch is needed. |
| `e4297d3bc2f` | `tt/generator_vllm.py` resolves `GEMMA4_31B_AUTOPORT_DIR` against `TT_METAL_HOME` instead of `os.getcwd()`. TTI launches the server from `vllm-tt-metal/src`, which previously made it look for `context_contract.json` under that directory. |
| `840b8301c40` | `--tensor-cache` on the three readiness runners (local convenience, not used by CI). |

## Recommended next steps, in priority order

1. **Add a `gemma-4-31b` entry to `model_performance_reference.json`** so blocks
   are graded. Until then a CI pass carries no performance meaning. Targets
   should be agreed rather than copied from our own run.
2. **Replace the blanket watchdog disable** with a raised timeout for this model,
   or warm the JIT cache during the image build, so hang detection is restored.
3. Run the `evals` and `release` lanes for coverage beyond throughput, knowing the
   Stage 11 accuracy gate is still blocked on canonical reference scores.
4. Exercise a near-advertised-context point so the batch ceiling of 3 at full
   context is tested rather than assumed.
5. Add a `"release": {"devices": ["P300X2"]}` block to `models-ci-config.json`
   only after the above.
