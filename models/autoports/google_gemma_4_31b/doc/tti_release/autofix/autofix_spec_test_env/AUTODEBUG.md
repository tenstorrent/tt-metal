# AutoDebug: vLLM parameter spec-test pytest isolation

## Scope and evidence reviewed

- Complete Stage 11 failure log: `tti_release_final3.log` (1,185 lines).
- `tt-inference-server-v2/test_module/llm_tests/vllm_param_conformance_test.py`.
- The chat-completions suite and its local `llm_module/conftest.py`.
- TTI and enclosing tt-metal pytest configuration.
- V2 workflow routing, release composition, and V2 run-script venv requirements.
- Inspection only: no implementation, hardware, server, eval cache, or release artifact was changed.

## Headline finding (P0, verified)

The failure is a pytest repository-isolation bug, not a missing TTI runtime dependency.

`VLLMParamConformanceTest._run_pytest_suite` launches the test file inside the TTI checkout but also passes an absolute `--output-path` located in the sibling Stage 11 `client_cache/tmp` tree. Pytest treats existing path-valued arguments as root-discovery inputs. The common ancestor of the suite path and this output path lies above `tt-inference-server`, so pytest selects `/localdev/odjuricic/tt-metal/pytest.ini` and walks up to `/localdev/odjuricic/tt-metal/conftest.py`. The dedicated V2 run-script venv correctly does not contain tt-metal's test stack, so collection stops at `from loguru import logger` with return code 4 before the TTI suite can write its parameter report.

This accounts for every observed symptom:

- The full release completed both Meta evals and all 17 benchmark points, then failed immediately when the parameter suite started.
- The failing command used the V2 run-script interpreter and an output directory outside the TTI checkout.
- The log reports no parameter JSON and exactly the enclosing tt-metal conftest import failure.
- `loguru` is absent from `.venv_v2_run_script`, but the suite's own import chain does not require it.

### Controlled contrasts

Using the same V2 interpreter and environment, collection produced these results:

1. Suite path without `--output-path`: exit 0, `rootdir` was the TTI checkout, `configfile` was its `pyproject.toml`, and 21 tests collected.
2. Suite plus the external absolute `--output-path`: exit 4 with the exact `tt-metal/conftest.py -> ModuleNotFoundError: loguru` failure.
3. Adding only `--confcutdir=<tti-root>`: collection passed, but pytest still reported the tt-metal root/config and an unknown tt-metal `timeout` option. This blocks the immediate conftest import but does not fully isolate TTI pytest semantics.
4. Adding only `--rootdir=<tti-root>`: still exit 4. Pytest documents and demonstrates here that `--rootdir` does not itself bound initial conftest discovery or select a config file.
5. Adding both explicit rootdir and confcutdir: collection passed, but pytest still selected the enclosing tt-metal `pytest.ini`; therefore this pair alone still inherits unrelated outer addopts/configuration.
6. Adding `-c <tti-root>/pyproject.toml`: exit 0, TTI root/config selected, 21 tests collected, and no outer conftest loaded.

## Correct intervention boundary

Fix the child pytest command construction in `vllm_param_conformance_test.py`; do not install `loguru` or the broader tt-metal test dependencies.

Minimal configuration-isolation fix:

```text
-c <_REPO_ROOT>/pyproject.toml
```

Recommended defensive form:

```text
-c <_REPO_ROOT>/pyproject.toml
--rootdir=<_REPO_ROOT>
--confcutdir=<_REPO_ROOT>
```

`-c` is the key argument because it pins the intended TTI pytest configuration and, in the verified contrast, also restores the intended TTI root. `--confcutdir` explicitly prevents future parent-conftest leakage. An explicit matching `--rootdir` makes the intended collection/report namespace clear, but it must not be used as the sole fix. If keeping the patch maximally small, use `-c` plus `--confcutdir`; adding rootdir is harmless defense and improves command self-documentation.

The existing `cwd=str(_V2_ROOT)` is already correct. Changing working directory alone cannot fix discovery because the two absolute path arguments still widen pytest's common ancestor. Installing `loguru` would merely advance into unrelated tt-metal fixtures, plugins, addopts, and dependencies while preserving the wrong test configuration.

## Required tests for the fix

### Focused static/unit tests

1. Extend `tt-inference-server-v2/tests/test_module/llm_tests/test_vllm_param_conformance_tests.py` to capture the child command and assert that it pins the TTI `pyproject.toml`, TTI rootdir, and TTI confcutdir (or the selected minimal subset), while retaining the suite, report, endpoint, and model arguments.
2. Run an isolation collection check with an existing output directory outside the TTI checkout. Expected: exit 0, TTI root/config, 21 collected tests, and no reference to `tt-metal/conftest.py`:

```bash
.workflow_venvs/.venv_v2_run_script/bin/python -m pytest \
  -c pyproject.toml \
  --rootdir=. --confcutdir=. \
  tt-inference-server-v2/llm_module/test_vllm_chat_completions.py \
  --output-path ../client_cache/tmp \
  --task-name vllm_chat_completions \
  --endpoint-url http://127.0.0.1:8000/v1/chat/completions \
  --model-name google/gemma-4-31B \
  --collect-only -o addopts='' -p no:cacheprovider
```

Use the real absolute Stage 11 client-cache path when reproducing; the relative spelling above is illustrative and must resolve outside `tt-inference-server`.

3. Focused unit suite:

```bash
.workflow_venvs/.venv_v2_run_script/bin/python -m pytest \
  -c pyproject.toml \
  tt-inference-server-v2/tests/test_module/llm_tests/test_vllm_param_conformance_tests.py -q
```

### Original failing check

After the focused checks pass, rerun only the V2 `spec_tests` workflow against the still-running exact autoport server, using the final3 runtime model spec, port 8000, and a fresh small output directory. This executes the same `VLLMParamConformanceTest` without rerunning Meta evals or 17 benchmark points:

```bash
.workflow_venvs/.venv_v2_run_script/bin/python tt-inference-server-v2/run.py \
  --model gemma-4-31B \
  --workflow spec_tests \
  --device p150x4 \
  --service-port 8000 \
  --server-url http://127.0.0.1 \
  --runtime-model-spec-json <final3-runtime-model-spec.json> \
  --output-dir <fresh-spec-test-output-dir>
```

Expected: LoggerForkSafetyTest and VLLMParamConformanceTest both produce blocks; pytest writes its parameter report; no return-code-4/no-report error occurs.

### Broader local regression

```bash
.workflow_venvs/.venv_v2_run_script/bin/python -m pytest \
  -c pyproject.toml \
  tt-inference-server-v2/tests/test_module/llm_tests \
  tt-inference-server-v2/tests/workflow_module/test_release_workflow.py -q
```

## Smallest release rerun strategy

The supported cheap verification is the standalone V2 `spec_tests` command above. It does not rerun completed evals or benchmarks.

However, current `ReleaseWorkflow` hard-codes the in-process child sequence `evals -> benchmarks -> spec_tests`, and its `BlockAccumulator` is populated in that process. No persisted resume/merge path was found that loads the final3 eval and benchmark blocks and replaces only the failed spec-test blocks in a new combined release report. Therefore:

- use standalone V2 `spec_tests` first to prove the repair cheaply;
- do not claim that standalone report retroactively changes the failed final3 combined report;
- if the handoff requires a newly generated single combined release report, the current workflow requires another full `--workflow release` run unless a separately reviewed resume/merge feature is implemented.

## Other hypotheses adjudicated

- **Missing `loguru` dependency:** refuted as the correct fix. It belongs to the enclosing tt-metal test harness, not the TTI parameter suite.
- **Wrong child working directory:** refuted. The runner already sets the V2 root, and the external absolute report path is sufficient to widen discovery.
- **`--rootdir` alone:** refuted by direct collection; outer conftest still loaded.
- **`--confcutdir` alone:** fixes the immediate import but leaves outer tt-metal pytest configuration active; incomplete isolation.
