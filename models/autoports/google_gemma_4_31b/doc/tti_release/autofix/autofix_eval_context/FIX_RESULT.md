# AutoFix Result: Gemma 4 release eval context and raw prompts

## Starting evidence

- Source: `autofix_eval_context/AUTODEBUG.md`.
- Original release symptom: lm-eval logged `Using max length 2048 - 1`
  because TTI omitted `max_length` from the local-completions model arguments.
- The prepared mandatory tasks also used Llama-Instruct-rendered
  `input_final_prompts`, despite the evaluated Gemma checkpoint being a base
  completion model with no chat template.
- Mandatory measured eval rows with neither a published nor same-prompt GPU
  reference were silently accepted as `NA`.

## Hypothesis experiments and fixes

### Exact lm-eval context propagation

- Hypothesis: deriving `max_length` from
  `DeviceModelSpec.max_context` fixes the 2048 fallback without changing prompt
  or output budgets.
- Verdict: verified.
- Fix: `tt-inference-server-v2/llm_module/eval_command.py` now adds
  `max_length=<max_context>` for text lm-eval adapters only when the task has no
  explicit override. It emits exactly `113280`, never `113281`, and emits
  nothing for a missing/non-positive context. It does not enable truncation or
  alignment.
- Focused checks cover exact 113280 propagation, explicit task override
  precedence, missing-context behavior, and the absence of `--limit`,
  `--samples`, and `--apply_chat_template`.

### Raw Gemma base prompts and cache validation

- Hypothesis: materializing local parquet tasks from source fields removes the
  embedded Llama wrappers while preserving task semantics and sample counts.
- Verdict: verified.
- Fix: added `workflows/prepare_gemma_meta_eval.py` and integrated it into the
  transactional Meta setup in `workflows/workflow_venvs.py`.
  - IFEval uses the original `input_question` for all 541 samples and retains
    its 1280-token output budget and original metrics.
  - GPQA reconstructs a reviewed plain-text reasoning/final-answer prompt from
    `input_question` and `input_choice_list` for all 448 samples. It retains the
    exact `best answer is ([A-Z])` extraction contract, gold answer, and
    2048-token output budget.
  - Stale `prompt`/`input_final_prompts` columns are excluded from the prepared
    mandatory task data. No Llama control marker occurs in any prepared prompt.
  - The prepared cache is accepted only when task IDs, local parquet files,
    exact 541/448 row counts, raw prompt columns, marker absence, output
    budgets, and the exact tokenizer context manifest all validate.
- Exact offline Gemma-tokenizer sweep of the rebuilt cache:
  - `meta_ifeval`: 541 samples, maximum prompt 346 tokens, maximum total
    1626 tokens (`346 + 1280`).
  - `meta_gpqa_cot`: 448 samples, maximum prompt 2440 tokens, maximum total
    4488 tokens (`2440 + 2048`).
  - Contract: `max_context=113280`, `truncation=false`, `alignment=null`.
- lm-eval `TaskManager` registers both rebuilt task IDs from the intended local
  YAML paths.

### Mandatory evals without baselines

- Hypothesis: a measured mandatory row with no published or GPU reference must
  form a blocker before known-issue matching, rather than silently count as
  `NA`.
- Verdict: verified.
- Fix: `tt-inference-server-v2/report_module/acceptance_criteria.py` now creates
  an explicit no-baseline blocker. Only an `EVALS` known issue matching the
  exact task demotes it to a visible waiver; a waiver for another task or
  workflow cannot match.
- The parent agent owns the runtime spec and will add separate narrow waivers
  for `meta_ifeval` and `meta_gpqa_cot`. The reason must state that no published
  or GPU Gemma-base reference exists for the exact raw-completion prompt
  contract, while retaining and reporting the measured row. No `-it` score is
  borrowed.

## Verification

- `python3 -m pytest -q tests/test_workflow_venvs_meta.py`
  - `9 passed`
- From `tt-inference-server-v2`:
  `python3 -m pytest -q tests/report_module/test_acceptance_criteria.py`
  - `36 passed`
- Broader root suites:
  `python3 -m pytest -q tests/test_workflow_venvs_meta.py tests/test_model_specification.py tests/test_model_catalog_yaml.py tests/workflows/test_acceptance_criteria.py`
  - `117 passed`
- Broader v2 suites:
  `python3 -m pytest -q tests/report_module tests/test_module/llm_tests/test_llm_eval_tests.py tests/test_run_external_spec.py`
  - `163 passed`, two pre-existing pytest collection warnings for the
    `TestStatus` enum.
- `python3 -m py_compile` passed for all changed implementation modules.
- `black` completed on all changed Python files.
- `git diff --check` passed.
- No hardware command, server request, container action, or commit was made by
  this fix agent.

## Final status

Fixed with source, focused-test, broader-suite, rebuilt-cache, exact-tokenizer,
and task-registration evidence. The next full no-Docker release rerun must
confirm the emitted commands contain `max_length=113280`, execute all 541/448
samples against the external autoport server, and render both measured rows
with their explicit task-specific no-baseline waivers.
