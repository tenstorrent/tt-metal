# Autofix Notes

Scope: TTI release workflow for `Qwen/Qwen3-4B` using `models/autoports/qwen_qwen3_4b`.

## Starting Evidence

The first TTI smoke against the already-running autoport vLLM server failed before issuing a benchmark request:

```text
ValueError: Model:=Qwen3-4B does not support device:=p150x4 in the 'prod' catalog
```

The runtime JSON spec was passed on the command line and contained the intended autoport wiring, but the v2 command path ignored it when building the workflow context.

Autodebug report: `diagnostics/AUTODEBUG.md`.

## Fix 1: V2 Runtime Spec Loading

Finding: `tt-inference-server-v2/workflow_module/command_factory.py::_build_context` always called `get_runtime_model_spec(model=args.model, device=args.device)`, which forced catalog lookup and rejected custom `p150x4` specs.

Change: when `args.runtime_model_spec_json` is set, load `ModelSpec.from_json(args.runtime_model_spec_json)` and use that spec for the v2 context.

Verification:

- Static command factory check loaded `model_id=id_tt-vllm-plugin_Qwen3-4B_p150x4_autoport_smoke`.
- Static check confirmed `impl.code_path=models/autoports/qwen_qwen3_4b`.
- Rerun smoke benchmark passed with `completed=1`, `failed=0`.

## Fix 2: Eval Context Length

Finding: the first full release attempt sent eval `model_kwargs.max_length=65536`, which exceeded the autoport context contract `40960`. This was a release harness/spec mismatch and should not be hidden by capping the server or benchmark context.

Change: `tt-inference-server-v2/llm_module/eval_command.py` now clamps eval `model_kwargs.max_length` to `device_model_spec.max_context` when a task requests a larger value.

Verification:

- Static command build produced `max_length=40960`.
- Final release log shows:
  - `Clamping r1_gpqa_diamond max_length: 65536 -> 40960`
  - `Clamping mmlu_pro max_length: 65536 -> 40960`
  - generated `lm_eval` commands use `max_length=40960`

## Final Blocker

After the harness fixes, release execution reached `lm_eval` dataset loading and failed on gated GPQA access:

```text
datasets.exceptions.DatasetNotFoundError: Dataset 'Idavidrein/gpqa' is a gated dataset on the Hub.
```

The available Hugging Face token could read dataset metadata, but `datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True)` still failed in the eval environment with the gated dataset error. This is an external auth/data-access blocker, not an observed model failure.

Recheck on 2026-07-03: `logs/gpqa_access_recheck_20260703T074315Z.log` reproduced the same failure in the exact TTI eval virtualenv with `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` present. `load_dataset` failed for plain environment loading, `token=True`, and an explicit token argument.

Local cache audit on 2026-07-03: `logs/gpqa_local_cache_audit_20260703T074550Z.log` found only the dataset card in the Hugging Face hub cache and no prepared GPQA dataset cache. No further autofix action was available without a token that can download the gated dataset files or a release-approved dataset mirror.
