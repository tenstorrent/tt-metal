# AutoFix Result: v2 external runtime model specs

## Starting evidence

- Diagnosis: `AUTODEBUG.md` in this directory.
- Pinned checkout: `tt-inference-server` at `v0.18.0`, SHA
  `d5913e816ac5dc33d86c1f3f5960348bc3fa4e2e` before stage edits.
- Original no-hardware failure: v2 `_build_context` ignored
  `--runtime-model-spec-json`, called the built-in catalog with
  `model=gemma-4-31B, device=p150x4`, and raised `ValueError` because the
  generated autoport model is intentionally not a stock catalog entry.
- The v2 parser independently rejected the same non-catalog model before it
  could read the supplied external spec. Eval selection used the outer CLI
  alias, and auth selection independently re-read the catalog.

## Hypothesis experiment

- Hypothesis: making `ModelSpec.from_json(...)` authoritative whenever an
  external runtime spec is supplied, while retaining catalog validation when
  it is absent, repairs the failure without adding or selecting a stock model.
- Prediction: parser admission, context construction, eval identity, and auth
  engine selection all preserve the loaded custom spec; the exact repro prints
  the base Gemma identity and `models/autoports/google_gemma_4_31b`.
- Verdict: **verified**.

## Fix

- `tt-inference-server-v2/run.py`
  - accepts a non-catalog `--model` only when a valid external runtime spec is
    supplied;
  - explicitly rejects absent/malformed specs and CLI/spec device mismatches;
  - preserves built-in model admission rules when no external spec is supplied;
  - lets tests pass argv directly to `parse_args` without changing normal CLI
    behavior.
- `tt-inference-server-v2/workflow_module/command_factory.py`
  - adds one shared `_resolve_model_spec` boundary;
  - loads flat legacy and combined external documents through
    `ModelSpec.from_json`;
  - uses the loaded model identity for eval lookup;
  - uses the same loaded spec for vLLM versus Forge/media auth behavior;
  - rejects a CLI/spec device mismatch rather than constructing an incoherent
    context.
- Focused regression coverage:
  - `tt-inference-server-v2/tests/test_run_external_spec.py` covers custom
    parser admission/rejection, missing/malformed documents, catalog fallback,
    and parser-to-command preservation of model and autoport implementation;
  - `tt-inference-server-v2/tests/workflow_module/test_command_factory.py`
    covers flat and combined formats, exact context/eval identity, device
    mismatch, and external vLLM/Forge/media auth branches.

Existing stage edits in `evals/eval_config.py`,
`test_module/server_tests_config.json`, and `test_module/test_suites/llm.json`
were not reverted or rewritten by this fix.

## Verification

Focused tests:

```text
PYTHONPATH=tt-inference-server-v2:$PWD pytest -q \
  tt-inference-server-v2/tests/workflow_module/test_command_factory.py \
  tt-inference-server-v2/tests/test_run_external_spec.py

48 passed in 0.41s
```

Neighboring v2 workflow regression suite:

```text
PYTHONPATH=tt-inference-server-v2:$PWD pytest -q \
  tt-inference-server-v2/tests/workflow_module \
  tt-inference-server-v2/tests/test_run_external_spec.py

110 passed in 0.59s
```

Exact no-hardware target repro after the fix:

```text
REPRO_FIXED gemma-4-31B models/autoports/google_gemma_4_31b P150X4 vLLM google/gemma-4-31B
PARSER_FIXED gemma-4-31B p150x4 /localdev/odjuricic/tt-metal/.exp_run/tti-release/gemma4-31b-20260716/autoport_smoke_spec.json
```

Static checks:

```text
python3 -m compileall -q <four changed Python/test files>  # pass
git diff --check                                           # pass
```

`ruff` is not installed in this pinned scratch checkout (`ruff: command not
found`), so no ruff result is claimed.

## Final status

**Fixed with focused CPU-only evidence.** No Tenstorrent hardware command and
no request to the live model server was made by this subtask.

Residual risk: the actual external-server TTI smoke is intentionally left to
the parent release workflow. That run remains necessary to validate the full
v1 bridge, provisioned v2 environment, real client request, and generated
report metadata together. The construction-level bridge and exact original
repro now pass and retain the generated autoport path.
