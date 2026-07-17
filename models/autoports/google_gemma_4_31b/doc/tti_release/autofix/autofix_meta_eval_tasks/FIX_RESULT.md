# Autofix result: Gemma Meta eval provisioning and embedded benchmark targets

## Result

The single comprehensive hypothesis in `AUTODEBUG.md` was confirmed and fixed
in the pinned TTI checkout at starting HEAD
`6ad299582d6fedeb3d98bb35be3f9109ff9d4d9d`. No hardware or inference-server
request was used by this repair.

The TTI harness now:

- keeps the evaluated and API identity as `google/gemma-4-31B`;
- explicitly selects `meta-llama/Llama-3.1-8B-Instruct-evals` only as the
  canonical Meta task/data preparation recipe;
- prepares exactly `meta_ifeval,meta_gpqa_cot` for this Gemma base model;
- fails closed on preparation failure or incomplete/wrong task artifacts,
  deletes partial staging state, restores cwd, and atomically publishes a
  validated work directory;
- preserves existing Llama 3.3-to-3.1 and vision-to-3.2-3B mappings;
- carries the runtime spec's exact `customer_functional`,
  `customer_complete`, and `customer_sellable` targets from the benchmark
  reference point through `LLMRunConfig`, the targeted benchmark block, and
  summary aggregation;
- grades that matching point with its embedded thresholds and tolerance,
  without consulting or modifying the stock central target JSON; and
- leaves generic sweep points informational and ungraded.

## Real preparation evidence

The existing authorized Hugging Face cache/credentials were sufficient for a
CPU-only cookbook preparation. `setup_evals_meta(...)` returned
`META_PREP_OK True`.

The prepared artifacts under
`.workflow_venvs/.venv_evals_meta/llama-cookbook/end-to-end-use-cases/benchmarks/llm_eval_harness/meta_eval/work_dir_gemma-4-31B`
include:

- `joined_ifeval.parquet` (613,131 bytes);
- `ifeval/ifeval.yaml` with `task: meta_ifeval`; and
- `gpqa_cot/gpqa_0shot_cot.yaml` with `task: meta_gpqa_cot` and canonical
  `meta-llama/Llama-3.1-8B-Instruct-evals` dataset identity.

An `lm_eval --tasks list_subtasks --include_path ...` registration probe listed
both exact task names. No `lm_eval` inference was run by the autofix agent.

Focused command-construction tests prove both full eval commands use
`model=google/gemma-4-31B`,
`base_url=http://127.0.0.1:8000/v1/completions`, omit
`--apply_chat_template`, and omit `--limit`.

## Full benchmark-config evidence

With `ONLY_BENCHMARK_TARGETS` explicitly unset and the real
`autoport_release_spec.json` loaded:

```text
COUNT 17
FIRST 128 128 1 8
MAX_TOTAL 65664 CONTRACT 113280
TARGET_TIERS ['customer_complete', 'customer_functional', 'customer_sellable']
TARGET_VALUES [('customer_functional', 1000.0, 20.0, 0.05),
               ('customer_complete', 700.0, 24.0, 0.05),
               ('customer_sellable', 600.0, 25.0, 0.05)]
GENERIC_WITH_TARGETS 0
```

Thus the complete 17-point sweep remains enabled, the intended reference point
is first, every request is within the 113,280-token context contract, and only
the matching reference point receives the exact embedded targets.

## Tests

- Meta mapping, failure cleanup, partial rebuild, task identity, and exact eval
  command tests under the plain root invocation (no manual `PYTHONPATH`):
  `8 passed`.
- Root neighboring workflow/benchmark tests together with focused Meta tests:
  `44 passed`.
- Focused v2 target propagation, runner, acceptance, summary, and aggregation
  tests: `72 passed`.
- Additional neighboring root tests: `36 passed`.
- Additional neighboring v2 LLM tests: `27 passed`.
- The target-plumbing implementation agent also ran a broader relevant v2
  suite: `299 passed`.
- `python3 -m black --check` on touched Python files and `git diff --check`
  passed.

No source commit was created by this autofix agent.
