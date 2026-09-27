# AutoFix: stage09 runtime prerequisite verification

Date: 2026-09-27. Final status: **blocked by missing serving dependencies and
the instruction not to install dependencies**. An externally provisioned
compatible runtime is required. No implementation changes were made.

## Starting evidence

- [AUTODEBUG.md](AUTODEBUG.md) identified missing `openai`, `uvloop`, the vLLM
  distribution, and TT plugin metadata in the existing Python environments.
- The original readiness `--help` failed with `ModuleNotFoundError: openai`.
- Source inspected: `/workspace/tt-metal/vllm`, commit
  `5ffebf4128f81ea5cf8413175eabde52cd8c8d75`.
- This bounded experiment pass followed `.agents/skills/autofix/SKILL.md` as
  the parent agent's dedicated hypothesis-verification subagent.

All commands ran from `/workspace/tt-metal`. Each Python subprocess used
`subprocess.run(..., timeout=30, capture_output=True, text=True)`. None timed
out. Only metadata and help commands were run; no hardware was accessed.

## Hypothesis experiments

### An existing interpreter already has the serving prerequisites

**Verdict: refuted for all four inspected interpreters.**

Executed each interpreter with `-B -c <probe>`:

```text
/workspace/tt-metal/python_env/bin/python
/opt/venv/bin/python
/usr/bin/python3
/usr/local/share/uv/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10
```

The exact probe code, complete argument vectors, exit codes, and outputs are
preserved in
[`autofix_metadata.json`](../../readiness_vllm/autofix_metadata.json).
It used `importlib.util.find_spec` for `openai`, `uvloop`, `vllm`, and
`vllm_tt_plugin`; `importlib.metadata.version` for their distributions; and
`importlib.metadata.entry_points` for both vLLM plugin groups.

All four probes exited 0 and found:

- `openai`, `uvloop`, and `vllm_tt_plugin`: absent.
- `vllm`: only a namespace module (`origin=None`), with no installed
  distribution version.
- `vllm.general_plugins` and `vllm.platform_plugins`: empty.

No fix was applied: changing the selected interpreter does not resolve these
missing prerequisites.

### The shared runner can parse arguments in the current environment

**Verdict: refuted; the missing-OpenAI diagnosis is verified.**

Exact command:

```bash
env PYTHONPATH=/workspace/tt-metal/models/common:/workspace/tt-metal /workspace/tt-metal/python_env/bin/python -B -m readiness_check.run_vllm_server --help
```

Exit code: **1**. The traceback resolves to the shared repository runner,
`models/common/readiness_check/run_vllm_server.py:97`, then fails at
`import openai` with `ModuleNotFoundError: No module named 'openai'`.

Evidence:
[`autofix_runner_help.log`](../../readiness_vllm/autofix_runner_help.log).
No fix was applied; a lazy import would only postpone the runtime failure.

### Exposing compatible vLLM and plugin source restores server startup

**Verdict: refuted; source exposure does not provide the dependency closure.**

Exact command:

```bash
env PYTHONPATH=/workspace/tt-metal/vllm:/workspace/tt-metal/vllm/plugins/vllm-tt-plugin/src:/workspace/tt-metal /workspace/tt-metal/python_env/bin/python -B -m vllm.entrypoints.openai.api_server --help
```

Exit code: **1**. Python reaches the actual server source at
`vllm/vllm/entrypoints/openai/api_server.py:17`, then fails at `import uvloop`
with `ModuleNotFoundError: No module named 'uvloop'`. A preceding
`vllm._version` warning is nonfatal; `uvloop` is the observed stopping error.

Evidence:
[`autofix_source_server_help.log`](../../readiness_vllm/autofix_source_server_help.log).
No fix was applied. This experiment added only the existing compatible source
directories for that subprocess; it did not assemble packages from caches.

## Final status

**Blocked.** The focused experiments independently confirm the AutoDebug
prerequisite diagnosis. Existing interpreters and source-path exposure cannot
execute even the two help commands. A source-only code patch cannot supply
the missing package closure and installed TT plugin metadata.

No dependencies were installed, no cache environment was assembled, and no
implementation was edited. The available next step is an externally
provisioned environment satisfying the exact requirements documented in
`AUTODEBUG.md`, followed by rerunning these same probes and help commands.
Serving, adapter correctness, device sampling, qualitative output, and
performance remain unverified.
