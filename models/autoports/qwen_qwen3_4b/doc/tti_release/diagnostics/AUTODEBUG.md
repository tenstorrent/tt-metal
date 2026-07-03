# AUTODEBUG: TTI v2 runtime-spec bypass failure

## Scope

Inspection-only diagnosis for the tiny no-Docker TTI benchmark smoke:

- Checkout: `/home/ubuntu/tti-release/qwen_qwen3_4b_20260703/tt-inference-server`
- Logged SHA: `3a7c6a021dac`
- Failure log: `/home/ubuntu/tti-release/qwen_qwen3_4b_20260703/logs/tti_smoke_benchmarks.log`
- Runtime spec: `/home/ubuntu/tti-release/qwen_qwen3_4b_20260703/autoport_smoke_runtime_model_spec.json`
- Copied runtime spec: `/home/ubuntu/tti-release/qwen_qwen3_4b_20260703/smoke_tti_cache/workflow_logs/runtime_model_specs/runtime_model_spec_2026-07-03_07-23-45_id_tt-vllm-plugin_Qwen3-4B_p150x4_autoport_smoke_3YHEbr9U.json`

No TT hardware commands, server operations, or implementation edits were performed. At inspection time the worktree already had an uncommitted modification in `tt-inference-server-v2/workflow_module/command_factory.py`; I did not create or revert it.

## Evidence-ranked findings

### 1. Confirmed root cause: v2 `_build_context` ignores `--runtime-model-spec-json` at `HEAD`

The failure happens before the benchmark client sends any request because the v2 command factory re-enters the prod catalog even though v1 already resolved and forwarded an external runtime spec.

Evidence:

- The log shows v1 honoring the supplied JSON first: `No validation is done, loading runtime model spec from JSON: .../autoport_smoke_runtime_model_spec.json`.
- The log then shows v1 saving a copied runtime spec and launching v2 with `--runtime-model-spec-json .../runtime_model_spec_2026-07-03_07-23-45_...json`.
- The traceback is entirely in v2 setup:
  - `tt-inference-server-v2/run.py:385` calls `CommandFactory.build(args)`.
  - `tt-inference-server-v2/workflow_module/command_factory.py:53` calls `_build_context(args)`.
  - `command_factory.py:100` calls `get_runtime_model_spec(model=args.model, device=args.device)`.
  - `workflows/model_spec.py:1300` raises `ValueError: Model:=Qwen3-4B does not support device:=p150x4 in the 'prod' catalog`.
- `git show HEAD:tt-inference-server-v2/workflow_module/command_factory.py` confirms the logged code shape at `HEAD`: `_build_context` unconditionally calls `get_runtime_model_spec(...)` before loading runtime config.
- A static contrast confirms the catalog failure is expected for this autoport-only spec:

```bash
PYTHONPATH=. python - <<'PY'
from workflows.model_spec import get_runtime_model_spec
try:
    get_runtime_model_spec(model="Qwen3-4B", device="p150x4")
except Exception as e:
    print(type(e).__name__ + ": " + str(e))
PY
```

Output:

```text
ValueError: Model:=Qwen3-4B does not support device:=p150x4 in the 'prod' catalog
```

Why this explains all observed symptoms:

- The supplied JSON is valid and contains `model_name=Qwen3-4B`, `device_type=P150X4`, `impl=tt-vllm-plugin`, and `inference_engine=vLLM`.
- The failure happens before request generation because `CommandFactory.build()` constructs metadata/context before `WorkflowRunner` executes any workflow task.
- The exact error string comes from `get_runtime_model_spec`, not from the benchmark tool or server.

### 2. 6e396b4 fixed the v1 validation side, but not this v2 context path

The commit named in the prompt, `6e396b43 Support external runtime specs in release workflows (#4345)`, changes v1 workflow validation/scripts and tests. It does not patch `tt-inference-server-v2/workflow_module/command_factory.py`.

Evidence:

- `git show --stat 6e396b4` lists files such as `workflows/validate_setup.py`, `workflows/run_workflows.py`, `benchmarking/run_benchmarks.py`, and tests, but no v2 command factory.
- Current v1 `run.py:725-757` has the correct precedence: if `args.runtime_model_spec_json` is present, it uses `ModelSpec.from_json(...)` and does not call `get_runtime_model_spec(...)`.
- The log confirms v1 got past this point and delegated to v2. The only remaining fatal catalog lookup is in v2.

### 3. Test gap: v2 command-factory tests explicitly exclude `_build_context`

`tt-inference-server-v2/tests/workflow_module/test_command_factory.py` documents that `_build_context` is out of scope. That leaves the runtime-spec bypass unpinned in exactly the helper that failed.

Evidence:

- The test module header says it covers arg-translation helpers and that the model-spec-dependent `_build_context` is out of scope.
- Existing tests cover `_load_runtime_config`, auth-token selection, and option construction, but not the invariant: with `runtime_model_spec_json` supplied, `_build_context` must not call `get_runtime_model_spec`.

## Smallest likely fix boundary

Patch only `tt-inference-server-v2/workflow_module/command_factory.py::_build_context`:

1. Import `ModelSpec` next to `get_runtime_model_spec`.
2. Load `runtime_config = _load_runtime_config(args.runtime_model_spec_json)` before selecting the model spec.
3. If `args.runtime_model_spec_json` is set, use `ModelSpec.from_json(args.runtime_model_spec_json)`.
4. Otherwise, keep the existing catalog path through `get_runtime_model_spec(...)`.

At inspection time, the worktree already contains exactly this uncommitted diff:

```diff
-from workflows.model_spec import get_runtime_model_spec
+from workflows.model_spec import ModelSpec, get_runtime_model_spec
...
-    model_spec, _, _ = get_runtime_model_spec(model=args.model, device=args.device)
+    runtime_config = _load_runtime_config(args.runtime_model_spec_json)
+    if args.runtime_model_spec_json:
+        model_spec = ModelSpec.from_json(args.runtime_model_spec_json)
+    else:
+        model_spec, _, _ = get_runtime_model_spec(model=args.model, device=args.device)
...
-    runtime_config = _load_runtime_config(args.runtime_model_spec_json)
```

Add a focused regression test in `tt-inference-server-v2/tests/workflow_module/test_command_factory.py` that:

- Creates or points to a combined runtime spec JSON.
- Monkeypatches `cf.get_runtime_model_spec` to raise.
- Calls `cf._build_context(...)` with `runtime_model_spec_json` set.
- Asserts the returned context uses the JSON model/device.

This test should live near `TestLoadRuntimeConfig` or in a new `TestBuildContextRuntimeSpec` section; the file's current "out of scope" comment should be updated accordingly.

## Cheap static validation run

I validated the candidate bypass against the current modified worktree with a pure Python import-level check. It does not contact a server or touch hardware:

```bash
PYTHONPATH=.:tt-inference-server-v2 python - <<'PY'
from argparse import Namespace
from pathlib import Path
from workflow_module import command_factory as cf
from workflows.workflow_types import DeviceTypes

spec = "/home/ubuntu/tti-release/qwen_qwen3_4b_20260703/smoke_tti_cache/workflow_logs/runtime_model_specs/runtime_model_spec_2026-07-03_07-23-45_id_tt-vllm-plugin_Qwen3-4B_p150x4_autoport_smoke_3YHEbr9U.json"

def boom(*args, **kwargs):
    raise AssertionError("catalog lookup should not be used when runtime_model_spec_json is supplied")

cf.get_runtime_model_spec = boom
ctx = cf._build_context(Namespace(
    runtime_model_spec_json=spec,
    model="Qwen3-4B",
    device="p150x4",
    num_prompts=None,
    output_dir=Path("/tmp/autodebug_v2_ctx"),
    workflow="benchmarks",
    service_port="8000",
    server_url="http://127.0.0.1",
))
assert ctx.model_spec.model_name == "Qwen3-4B"
assert ctx.device == DeviceTypes.P150X4
print("ok: _build_context used runtime_model_spec_json without catalog lookup")
PY
```

Output:

```text
ok: _build_context used runtime_model_spec_json without catalog lookup
```

## Other potential issues / follow-ups

- `tt-inference-server-v2/run_agentic.py::_ensure_agentic_venv` still calls `get_runtime_model_spec(model=args.model, device=args.device)` and does not parse `--runtime-model-spec-json`. That is not on the failing benchmark path, but it is another v2 launcher runtime-spec bypass gap for external agentic specs.
- `tt-inference-server-v2/workflow_module/command_factory.py::_resolve_auth_token` still probes the catalog to decide forge/media versus vLLM auth, catches failures, and defaults to the JWT path. This is harmless for the reported external vLLM smoke with auth disabled, but an external forge/media runtime spec could select the wrong bearer-token style unless this helper also reads `ModelSpec.from_json(...)` when available.
