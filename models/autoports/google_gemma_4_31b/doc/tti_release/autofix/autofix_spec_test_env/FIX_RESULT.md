# AutoFix Report: vLLM parameter spec-test pytest isolation

## Starting Evidence

- Source diagnosis: `AUTODEBUG.md` in this directory, read in full before editing.
- Original release symptom: the nested vLLM parameter pytest process accepted an absolute suite path inside the TTI checkout and an absolute report path in the sibling Stage 11 `client_cache/tmp` tree. Pytest widened discovery to the enclosing tt-metal checkout, imported `tt-metal/conftest.py`, and failed on the unrelated missing `loguru` dependency before producing a parameter report.
- Verified diagnosis contrasts established that `cwd`, `--rootdir`, or `--confcutdir` alone were insufficient. Pinning the TTI `pyproject.toml` with `-c`, together with matching `--rootdir` and `--confcutdir`, selected the intended TTI configuration and bounded conftest discovery.

## Hypothesis Experiment

- Hypothesis: the failure is caused by unbounded child-pytest configuration/root discovery when an external absolute output path is present.
- Prediction: adding `-c <TTI root>/pyproject.toml`, `--rootdir <TTI root>`, and `--confcutdir <TTI root>` to the existing child command will retain all runtime inputs while collecting the suite under the TTI root/config and avoiding the enclosing tt-metal conftest.
- Focused experiment: added a subprocess-capture unit regression whose report directory is an existing absolute path outside the TTI checkout. The test verifies the complete child command, including suite, output, task, endpoint, model, and quiet arguments; it also verifies the unchanged V2 `cwd`, leading `PYTHONPATH` entries, and report consumption.
- Result: verified. Focused unit suite passed 4/4.
- Controlled reproduction: using the V2 run-script interpreter, the real absolute Stage 11 `client_cache/tmp` output directory, and `--collect-only` selected:
  - rootdir: the TTI checkout;
  - configfile: `pyproject.toml` in the TTI checkout;
  - collected tests: 21;
  - exit code: 0;
  - no enclosing `tt-metal/conftest.py` import.
- Verdict: verified and fixed.

## Fix

- Updated `tt-inference-server-v2/test_module/llm_tests/vllm_param_conformance_test.py` to prepend the robust repository boundary to every nested parameter-suite pytest command:
  - `-c <_REPO_ROOT>/pyproject.toml`
  - `--rootdir <_REPO_ROOT>`
  - `--confcutdir <_REPO_ROOT>`
- Preserved the existing interpreter, suite path, output path, task name, endpoint URL, model name, quiet flag, environment, and V2 working directory.
- Added `test_run_pytest_suite_isolates_config_with_external_output_path` to `tt-inference-server-v2/tests/test_module/llm_tests/test_vllm_param_conformance_tests.py`.
- No dependency was added. No server, hardware, evaluation cache, or live endpoint test was touched. Changes are intentionally uncommitted for main-agent review.

## Verification

### Focused unit regression

```text
.workflow_venvs/.venv_v2_run_script/bin/python -m pytest -c pyproject.toml \
  tt-inference-server-v2/tests/test_module/llm_tests/test_vllm_param_conformance_tests.py -q
```

Result: `4 passed` (one existing `TestConfig` collection warning).

### Controlled external-output collection

```text
.workflow_venvs/.venv_v2_run_script/bin/python -m pytest \
  -c <TTI_ROOT>/pyproject.toml \
  --rootdir <TTI_ROOT> --confcutdir <TTI_ROOT> \
  <TTI_ROOT>/tt-inference-server-v2/llm_module/test_vllm_chat_completions.py \
  --output-path <STAGE_ROOT>/client_cache/tmp \
  --task-name vllm_chat_completions \
  --endpoint-url http://127.0.0.1:8000/v1/chat/completions \
  --model-name google/gemma-4-31B \
  --collect-only -o addopts='' -p no:cacheprovider
```

Result: exit 0, TTI root/config selected, `21 tests collected` in 0.02 seconds.

### Broader local regression

```text
.workflow_venvs/.venv_v2_run_script/bin/python -m pytest -c pyproject.toml \
  tt-inference-server-v2/tests/test_module/llm_tests \
  tt-inference-server-v2/tests/workflow_module/test_release_workflow.py -q
```

Result: `72 passed` (two existing collection warnings).

### Formatting and diff checks

```text
black --check \
  tt-inference-server-v2/test_module/llm_tests/vllm_param_conformance_test.py \
  tt-inference-server-v2/tests/test_module/llm_tests/test_vllm_param_conformance_tests.py
git diff --check
```

Result: both files left unchanged by Black check; `git diff --check` passed. Ruff is not installed in the V2 run-script environment.

## Final Status

- Fixed locally with focused and broader evidence.
- The remaining required confirmation is the live standalone V2 `spec_tests` workflow against the exact autoport server. That endpoint exercise was explicitly left to the main agent.
- Working-tree scope: two modified TTI source/test files plus this report outside the nested checkout; no commit created.
