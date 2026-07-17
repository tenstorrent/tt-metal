# AutoFix Result: Gemma Stage 11 IFEval schema

## Starting evidence

- Diagnosis: `AUTODEBUG.md` in this directory.
- Original failure: the complete 541-sample IFEval generation reached the copied
  Meta scorer, then `ifeval/utils.py::process_results` raised `KeyError: 'prompt'`.
- Focused verification: the staged parquet had `input_question`, `key`,
  `instruction_id_list`, and `kwargs`, but no `prompt`; the copied scorer reads
  `doc["prompt"]` unconditionally and uses it for prompt-dependent instructions.
- Verdict: verified deterministic preparation/cache-validation contract bug.

## Fix

Changed only the scratch TTI checkout requested by the parent agent:

- `workflows/prepare_gemma_meta_eval.py`
  - Replaces the published Llama-rendered prompt with `prompt = input_question`.
  - Keeps `input_final_prompts` removed.
  - Generates IFEval from `doc_to_text: prompt`, so inference and scoring use one
    authoritative raw string.
  - Requires `key`, a non-empty `instruction_id_list`, aligned `kwargs` mappings,
    and an `nth_paragraph` key in every kwargs mapping before publishing data.
  - Preserves the exact 541 sample count, raw-prompt marker checks, 1280-token
    IFEval output budget, tokenizer context validation, and 113280 context contract.
- `workflows/workflow_venvs.py`
  - Requires `prompt`, `input_question`, `key`, `instruction_id_list`, and
    `kwargs` in the cached IFEval parquet.
  - Requires `prompt == input_question`, non-empty raw prompts, no Llama control
    markers, non-null keys, valid instruction IDs, aligned list metadata,
    mapping-compatible kwargs, and the scorer-indexed `nth_paragraph` key.
  - Requires `doc_to_text: prompt`; it no longer certifies a cache that rejects
    the scorer-required prompt column.
- `tests/test_workflow_venvs_meta.py`
  - Updates the valid cache fixture to the scorer-complete schema.
  - Adds preparation regression coverage proving the rendered prompt is removed
    and rebuilt from `input_question` for all 541 rows.
  - Adds negative cache tests for missing fields, mismatch, Llama markers,
    empty/unaligned metadata, non-mapping kwargs, and missing `nth_paragraph`.
  - Adds a staged-scorer contract test that invokes the copied
    `ifeval/utils.py::process_results` in the Meta eval venv and asserts all four
    required metric keys are returned for a schema-complete row.

The copied scorer, task sample counts, generation budgets, context limits, GPQA
recipe, server/model code, and shared prepared cache were not changed.

## Verification

- `python -m pytest -q tests/test_workflow_venvs_meta.py`
  - `21 passed in 7.69s`
- `python -m pytest -q tests/test_workflow_venvs_meta.py tests/test_workflows.py`
  - `50 passed in 8.54s`
- `python -m pytest -q tests/test_workflow_venvs_meta.py::test_staged_ifeval_scorer_accepts_prepared_schema`
  - `1 passed in 7.18s`
  - Exercises the staged copied scorer and verifies
    `prompt_level_strict_acc`, `inst_level_strict_acc`,
    `prompt_level_loose_acc`, and `inst_level_loose_acc` are all returned.
- Read-only validation of the existing 541 source rows confirmed every key is
  non-null, ID/kwargs lists are aligned, every kwargs item converts from Arrow
  to a mapping, and every mapping contains `nth_paragraph`.
- `python -m black --check workflows/prepare_gemma_meta_eval.py workflows/workflow_venvs.py tests/test_workflow_venvs_meta.py`
  - `3 files would be left unchanged`
- `git diff --check`
  - passed with no output.

## Final status and residual risk

Fixed with focused and broader evidence. The shared prepared cache intentionally
still has the old schema; the parent agent must rebuild it through the reviewed
preparation helper, validate 541 `prompt == input_question` rows and the four-key
scorer probe, then run a fresh release evaluation. No completed-generation cache
exists to reuse safely. No hardware/server action was taken and no commit was
created.
