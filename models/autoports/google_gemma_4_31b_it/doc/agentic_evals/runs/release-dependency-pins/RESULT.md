# Release dependency pin findings

**Immutable TTI publication commits currently do not freeze Harbor. No supported model-entry Harbor revision field was found.** Mini-swe-agent itself is already explicitly versioned 2.2.8, but its transitive tool environment is not fully locked. Read-only investigation; zero device hours and no source changes.

## Stock source behavior

In the separate TTI checkout, workflows/workflow_venvs.py:46 sets HARBOR_REF="tt-inference-server". setup_evals_agentic at line320 accepts model_spec but does not read it; it always invokes checkout_pinned_repo(harbor_dir,HARBOR_REPO,HARBOR_REF), then `uv pip install -e '<harbor_dir>[kubernetes]'`. requirements/evals-agentic.txt explicitly excludes Harbor because this hook installs it.

checkout_pinned_repo at line84 is converging, not cache-pinning: on every setup it sets origin, fetches `--depth 1 origin <ref>`, and forcibly checks out detached FETCH_HEAD. Cached runner venvs/checkout directories do not freeze the branch; they fetch its current head. VenvConfig.setup at line282 skips only creation of an existing venv; requirements installation and setup_function still run. workflow_module/commands.py:542 calls provision for the workflow and dependency venvs; workflows/venv_provisioner_provider.py:40 delegates to that setup.

HarborEvalConfig has agent version/env/config fields, dataset_ref, task lists and resource controls, but no host Harbor revision field. HARBOR_REF is a literal, not an environment lookup. Model/device env registration therefore cannot override it. An agent_env UV_OVERRIDE concerns the task-side installation environment and cannot change the earlier host checkout. Installing a separate Harbor wheel by constraints would not prevent the hook's local editable checkout either.

reference_config/evals/eval_config.py:356 pins MINI_SWE_AGENT_VERSION="2.2.8", and this model passes it in SWE agent_kwargs. The locally inspected stock Harbor tree e76c762fca5267dfb36e1f900f0a7724ed72e735 installs `mini-swe-agent==<version>` inside the task container, in src/harbor/agents/installed/mini_swe_agent.py:761-786. That install additionally requests unversioned litellm[mcp], orjson and fastapi; Python/tool installer versions and transitive package resolution are not established by the mini2.2.8 pin alone. Terminus-2 code is part of the Harbor checkout, so pinning Harbor freezes its source version too.

## Available run evidence and limits

Local parser-protocol/result.json identifies the inspected Harbor SHA e76c762fca5267dfb36e1f900f0a7724ed72e735 and explicitly says actual CI fetch must be verified. It is not a CI version measurement. Collected Terminal premodel-failure logs (bringup/artifacts/stage12-terminal-job.log and smoke2-terminal_bench_2_1/collected/.../run_logs) show the TTI bb321bf checkout but no Harbor checkout/setup result; the model failed before a task run. They cannot establish a CI Harbor SHA or installed mini version. No actual Harbor git HEAD/direct_url or mini tool-version metadata was found in the currently collected smoke artifacts searched. Ongoing CI runs were neither interrupted nor independently polled/dispatched by this effort.

## Minimal freeze path

A model-entry-only solution is unavailable under the current stock API. Proposed generic, default-preserving repair on a defect-named branch: let setup_evals_agentic read an optional model metadata `harbor_git_ref`, require a full immutable commit SHA when supplied, otherwise retain stock HARBOR_REF; checkout through the same helper and log verified `git rev-parse HEAD`. The model's publication entry can then set the exact Harbor SHA actually validated by dry run. This changes dependency selection only, not the task runner, sandbox, task selector or release route, but it is still a generic harness repair and must follow the stage's generic-branch/release-route-repair rules. Do not silently patch the shared constant in a model branch or mislabel a mutable ref as frozen.

For complete package reproducibility beyond Harbor/mini source versions, preserve actual dry-run package manifests and constrain the host and task-side dependency sets through a generic supported lock mechanism. The current setup uses editable uv pip install rather than uv sync --frozen, so the repository's uv.lock alone is not proof that dependencies were locked. A repaired Harbor SHA pin plus unchanged mini2.2.8 does not justify claiming all transitive packages frozen. Exact mechanisms and confirmation tests for transitive constraints need a separate implementation effort if full environment identity is required.

Before a release: record dry-run Harbor resolved SHA, actual mini version, host/task package versions and source pins; prove the release resolves identically. Same machine or same TTI SHA does not substitute for this check. Until the pin capability exists or an explicitly accepted alternative freeze contract is provided, Harbor source freeze is an unresolved dependency-registration requirement, not a measured model failure.

## Concrete local repair prepared

Local defect branch `fix/model-scoped-harbor-revision`, worktree `/home/mvasiljevic/gemma4-31b-pipeline-v3/tti-harbor-pin-repair`, commit `7079fe2` on publication base8437f186a26dffda66b263a079c777f0670f15c4. Source-only patch: harbor-revision-pin.patch alongside this result. No push or active-publication edit.

Files: workflows/workflow_venvs.py and tests/workflows/test_harbor_revision_pin.py. setup_evals_agentic accepts optional model metadata.harbor_git_ref, rejecting anything except a full40hex SHA before checkout. SHA case normalizes lowercase. Absent metadata preserves stock tt-inference-server branch selection. After the unchanged checkout helper, a bounded read-only git rev-parse HEAD verifies the exact requested commit and logs requested/resolved references before editable install; mismatches/errors fail closed. No task-runner, selector or sandbox behavior changed.

Confirmation: 49 focused tests passed across test_harbor_revision_pin.py and existing test_eval_config_harbor_env.py, including default selection, exactSHA forwarding, invalid/mutable refs, verification failure/mismatch and no installation on failure. Ruff format/check and git diff --check passed. Tests mock checkout/install and do not contact devices/network. Full repository hooks subsequently passed in the detached repair worktree; see final hook confirmation below.

Use `git am <absolute path>/harbor-revision-pin.patch` or cherry-pick the local commit after reviewing generic-repair publication authorization. Model registration remains deliberately absent until actual CI-resolved Harbor SHA is established. This repair freezes Harbor source when configured, including terminus code; it does not lock mini's transitive package set, as noted above.

## Full hook confirmation

All hooks passed with inherited tt_metal_home and TT_METAL_HOME unset: Ruff, format, full pytest, copyright. Final log: bringup/artifacts/stage12-harbor-pin-hooks-final.log. Initial full run exposed only three new-test logging capture failures (3756 passed,15 skipped): prior suite setup disables logger propagation, so caplog saw no message despite correct stdout. Changed only the new test to assert the logger.info call directly; production default behavior unchanged. Amended local commit is7079fe20b4a7c0eb8dc441fd0e0bc455bafda97c; patch refreshed. AUTODEBUG.md and AUTOFIX.md record the contract and repair. No publication or active checkout edit.
