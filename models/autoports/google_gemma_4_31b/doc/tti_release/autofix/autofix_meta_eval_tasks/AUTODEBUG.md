# AutoDebug: Gemma 4 31B TTI Meta-eval provisioning and benchmark selection

## Scope and result

Inspection-only diagnosis of the first full no-Docker TTI release run at TTI
HEAD `6ad299582d6fedeb3d98bb35be3f9109ff9d4d9d`. No hardware or live-server
requests were made and no source was edited.

The two missing mandatory rows have a complete, source-proven causal chain:
TTI treats the evaluated Hugging Face model identity as the Meta-eval recipe
identity, asks `llama-cookbook` for the nonexistent
`google/gemma-4-31B-evals` collection, ignores the preparation failure, and
then passes a broken include path to `lm_eval`. Both exact task names therefore
fail registration before any dataset sample or server request is attempted.

The smallest methodology-preserving repair is to keep the evaluated model and
API model as `google/gemma-4-31B`, but explicitly map its Meta-eval *recipe* to
the canonical `meta-llama/Llama-3.1-8B-Instruct-evals` recipe. This produces the
real `meta_ifeval` and `meta_gpqa_cot` definitions instead of relabeling generic
lm-eval tasks. The release config must continue to use local completions with
`apply_chat_template=False`, and the run must have no `--limit`.

For benchmarks, the observed 17 points are the documented full-release
behavior. `ONLY_BENCHMARK_TARGETS=1` is a supported switch, but it explicitly
turns off the generic sweep and is therefore limiting; it should remain unset
for the full release. An independent target-reporting defect also needs repair:
v2 currently drops the custom spec's embedded target values and grades against
the central reference JSON, where this base Gemma/P150X4 pair is absent.

## Evidence-ranked findings

### 1. Root cause: EVALS_META conflates evaluated model identity with the supported Llama recipe identity (certain)

- `evals/eval_config.py:3715-3754` correctly identifies the evaluated model as
  `google/gemma-4-31B`, configures the exact rows `meta_ifeval` and
  `meta_gpqa_cot`, uses `local-completions`, and explicitly leaves
  `apply_chat_template=False`.
- `workflows/workflow_venvs.py:302-313` derives `_model_name` directly from
  `model_spec.hf_model_repo`, then writes
  `evals_dataset = f"{_model_name}-evals"`. For this model that is
  `google/gemma-4-31B-evals`.
- The pinned cookbook's `prepare_meta_eval.py:15-31,280-288` accepts only the
  Llama 3.1/3.2 eval collections. It has no generic model path. Its IFEval
  builder is even narrower (`prepare_meta_eval.py:35-53`): it accepts named
  Llama 3.1/3.3 Instruct recipes and loads a `meta-llama/...-evals` dataset.
- The run log confirms those exact values and failure:
  `../tti_release.log:768-774` prepares `work_dir_gemma-4-31B`, logs
  `model_name: google/gemma-4-31B`, then rejects the dataset as outside the
  Llama 3.1/3.2 collection.

This is not an authentication failure, a missing Hugging Face cache, a server
failure, or a Gemma implementation failure. The cookbook rejects the name
locally before attempting the task dataset load.

Why the canonical Llama recipe is the correct intervention boundary:

- `prepare_meta_eval.py:168-220` copies the canonical Meta task templates and
  rewrites their dataset/template recipe from the chosen eval collection.
- `prepare_meta_eval.py:234-250` prepares the recipe datasets; the actual
  inference model is not selected here.
- The inference identity is independently constructed later by
  `tt-inference-server-v2/llm_module/eval_command.py:299-323` from
  `model_spec.hf_model_repo`. The failing run proves it remains
  `model=google/gemma-4-31B` and uses `/v1/completions`
  (`../tti_release.log:795-796,815-816`).

Therefore a recipe mapping does not substitute a Llama model for the generated
Gemma autoport. It only selects the canonical dataset/task preparation recipe.
Renaming installed generic `ifeval` or `gpqa_main_cot_zeroshot` tasks would make
rows appear, but would not be the exact Meta tasks and is not recommended.

### 2. Root cause: Meta dataset preparation failure is deliberately ignored, leaving broken task include paths (certain)

- `workflows/workflow_venvs.py:319-326` calls the cookbook with non-throwing
  `run_command`, logs a warning on nonzero return, but does not set
  `setup_succeeded=False` or return failure.
- `VenvConfig.setup` only rejects a setup hook if it returns false
  (`workflows/workflow_venvs.py:146-149`). The stale true value returned at
  `workflow_venvs.py:335` lets v2 launch the release.
- At command-build time, `eval_command.py:349-365` unconditionally creates a
  per-invocation symlink to the presumed model work directory. Both observed
  symlinks exist but their targets do not.
- Consequently `lm_eval` receives an include path but sees no task YAML. The
  two subprocesses fail with `Tasks were not found: meta_ifeval` and
  `Tasks were not found: meta_gpqa_cot`
  (`../tti_release.log:795-809,815-829`). No model request is made.
- v2 correctly converts both subprocess failures into visible FAIL blocks
  (`tt-inference-server-v2/test_module/llm_tests/llm_eval_tests.py:214-233,
  318-350`); the misleading `workflow.evals: ... blocks=2` log line records
  block production, not eval success.

There is a related stale-partial risk: the setup guard checks only whether
`work_dir_<model>` exists (`workflow_venvs.py:290-291`). If preparation fails
after the cookbook starts copying files, a later run can skip preparation even
when a parquet or task YAML is missing. An existence-only cache is not a valid
completion condition.

### 3. Expected behavior: full release appends the generic benchmark sweep to the custom target point (certain)

- `benchmarking/benchmark_config.py:542-568` reads the resolved runtime spec's
  `device_model_spec.perf_reference`; `:571-593` makes it the first benchmark
  task. `get_benchmark_config` explicitly does not fall back to the import-time
  catalog (`:700-708`). Thus runtime-spec selection precedence is correct.
- Unless `ONLY_BENCHMARK_TARGETS` is nonempty, the default text sweep is
  appended (`:594-673`) from the ISL/OSL table at `:105-116`.
- v2 flattens and de-duplicates identical `(isl, osl, concurrency,
  num_prompts)` points (`tt-inference-server-v2/llm_module/benchmark_configs.py:54-73`).
  The custom `(128,128,c1,n8)` target duplicates the first generic point, so
  the result is 17 unique points, not 18. The run logs exactly 17 and starts
  with that target point (`../tti_release.log:839-844`).
- The benchmark documentation says `ONLY_BENCHMARK_TARGETS` will “turn off
  benchmark sweeps” (`benchmarking/README.md:156-165`), and the test helper
  explicitly unsets it when sweeps are expected
  (`tests/test_benchmark_config.py:34-45`). Any nonempty value, including `0`,
  disables sweeps.

Verdict: do not set `ONLY_BENCHMARK_TARGETS` on the full release rerun. Doing so
would preserve the configured target point and its eight prompts, but suppress
16 unique characterization points; it is not non-limiting. It remains suitable
for the already-completed tiny smoke or for an explicitly approved target-only
workflow.

### 4. Independent impending blocker: v2 discards the runtime spec's embedded performance targets (certain)

- `BenchmarkTaskParams.targets` survives into benchmark construction, but
  `tt-inference-server-v2/llm_module/benchmark_configs.py:66-72` creates
  `LLMRunConfig` with only ISL, OSL, concurrency, and prompt count.
- `tt-inference-server-v2/workflow_module/summary_report.py:127-143` grades the
  resulting aggregate by calling `get_performance_targets(model, device)`.
- That helper reads only the central
  `benchmarking/benchmark_targets/model_performance_reference.json` (or an
  `OVERRIDE_BENCHMARK_TARGETS` file) and returns all-None targets when the
  model/device is absent
  (`tt-inference-server-v2/test_module/_test_common/target_check.py:23-30,
  78-127`). The central file has `gemma-4-31b-it/P300X2`, not
  `gemma-4-31B/P150X4`.
- The schemas also differ: the custom spec carries explicit
  `customer_functional`, `customer_complete`, and `customer_sellable` tiers,
  while the central loader reads only `targets.theoretical` and derives tiers
  with fixed multipliers (`target_check.py:33-75`). Merely copying the existing
  object unchanged into the central JSON will not make v2 read it.

This does not explain the missing Meta tasks, but it contradicts the required
custom-target handoff and will otherwise produce missing/NA benchmark target
checks after the long sweep.

## Smallest comprehensive fix plan

1. In `setup_evals_meta`, separate the evaluated model from the preparation
   recipe. Add an explicit, reviewable mapping for
   `google/gemma-4-31B -> meta-llama/Llama-3.1-8B-Instruct-evals` (do not apply a
   silent fallback to every unknown model). Keep `config["model_name"]` as the
   evaluated Gemma identity for provenance, set only `config["evals_dataset"]`
   to the canonical recipe, and set the preparation task list to the two exact
   configured rows (`meta_ifeval,meta_gpqa_cot`) so full IFEval is prepared
   without needlessly preparing unrelated MATH tasks.
2. Make preparation transactional and fail closed: restore cwd with
   `try/finally`; on nonzero preparation return remove the partial model work
   directory and return false; do not launch v2. On success validate at least
   the two task YAMLs and `joined_ifeval.parquet` (or write/check a completion
   marker only after validation). An existing directory without those artifacts
   must be rebuilt.
3. Preserve the current eval contract: `google/gemma-4-31B`,
   `local-completions`, `/v1/completions`, `apply_chat_template=False`, exact
   task names, full datasets, no `--limit`, and device context 113280. Do not
   replace the tasks with generic lm-eval aliases and do not borrow an `-it`
   accuracy baseline.
4. Leave `ONLY_BENCHMARK_TARGETS` unset for the full release. Keep the runtime
   spec's target point first, then the complete valid sweep under max context.
5. In a focused follow-up, preserve embedded benchmark target tiers through
   `LLMRunConfig`/blocks and make summary grading prefer those explicit runtime
   targets for the matching point. This is cleaner than adding a duplicate
   central entry and avoids changing explicit customer tiers into the central
   fixed-multiplier schema. If a short-term override is unavoidable, generate a
   small `OVERRIDE_BENCHMARK_TARGETS` file with an explicitly documented schema
   conversion and verify it matches the runtime spec; do not silently grade NA.

## Verification plan

Cheap/local gates before using the live server:

1. Unit-test recipe resolution: Gemma evaluates as
   `google/gemma-4-31B` while its dataset recipe resolves to exactly
   `meta-llama/Llama-3.1-8B-Instruct-evals`; existing Llama remaps (3.3 to 3.1,
   vision to 3.2-3B) remain unchanged; unknown non-Llama EVALS_META models fail
   with an actionable error.
2. Unit-test setup failure: mocked cookbook nonzero return makes
   `VenvConfig.setup` fail, removes partial output, and restores cwd. A partial
   directory without required artifacts is rebuilt rather than accepted.
3. After real authenticated preparation, run `lm_eval --tasks list_subtasks
   --include_path <prepared-work-dir>` and require both exact names. Inspect the
   prepared YAML/data: `meta_ifeval`, `meta_gpqa_cot`, canonical Llama recipe,
   full data, and no sample limit.
4. Build both eval commands without executing them. Require
   `model=google/gemma-4-31B`, port 8000, `/v1/completions`, no
   `--apply_chat_template`, no `--limit`, and the prepared include path.
5. Benchmark config regression: with `ONLY_BENCHMARK_TARGETS` unset, the current
   custom spec yields 17 unique text points and begins with
   `(128,128,c1,n8)`; with it set, it yields one point. The full release test
   must use the unset case. Also test that the exact custom target tiers reach
   the matching report block and produce non-NA target checks.

Live rerun gates:

6. Run the two full Meta tasks. Require subprocess return 0, result JSON with
   exact task keys and expected metrics (`prompt/inst strict/loose` for IFEval,
   `exact_match,strict-match` for GPQA), and two non-FAIL report rows. No waiver
   and no dataset limit.
7. Run the full release without `ONLY_BENCHMARK_TARGETS`. Require all 17 valid
   benchmark points (no point with ISL+OSL over 113280), the custom target row's
   explicit checks, and no missing/NA target metrics caused by lookup plumbing.

## Claim review / ruled-out alternatives

- The external server URL is healthy at the corrected port in the failure log
  (`../tti_release.log:794`); the missing tasks occur entirely before inference.
- The exact eval commands already preserve the base completion path and omit
  chat-template application. Changing the model/server adapter cannot register
  missing YAML tasks.
- The context contract is not involved: both task registration failures occur
  before tokenization or request validation.
- Generic lm-eval `ifeval` and `gpqa_main_cot_zeroshot` are installed and
  register successfully, but renaming them would change the requested Meta
  methodology. They are useful only as evidence that the venv itself is
  functional, not as the release fix.
- `ONLY_BENCHMARK_TARGETS=1` does not fix any eval problem and would hide most
  of the documented full-release benchmark sweep.
