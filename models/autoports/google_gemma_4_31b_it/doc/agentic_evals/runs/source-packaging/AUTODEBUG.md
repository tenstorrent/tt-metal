# AutoDebug: source-only deployment blockers

Inspected 2026-11-06, checkout HEAD `b030d0b10a88ec05933bf95bc106b830f5166b3b` (existing working-tree changes preserved). Inspection only: no model/device execution, source edits, plugin edits, publication, or prior Gemma implementation reads. This report intentionally lives here; root AUTODEBUG.md is preserved.

## Headline findings

Two independent, deterministic packaging blockers exist under the stated deployment conditions. The first encountered is the missing readiness package during adapter import; after that is repaired, default model construction still requires an excluded documentation artifact. Neither calls for numerical, scheduler, trace, or shared-plugin runtime changes.

### 1. Adapter import requires the installed skills runtime

Direct observations:

- `tt/generator_vllm.py:13` imports `Gemma4Generator` at module import time.
- `tt/generator.py:12` unconditionally imports `Generator` from `readiness_check.contract`; line 24 subclasses it.
- The contract exists in the installed model-bringup plugin, not the autoport runtime sources. With no `readiness_check` on CI's Python path, this import raises `ModuleNotFoundError` before adapter construction.

Diagnosis: the adapter's runtime import graph accidentally includes a development/readiness package. Local success with the installed runtime on PYTHONPATH does not establish standalone publication correctness. Missing CI availability is the scenario supplied by the requester, not a remotely observed CI failure in this investigation.

Required contract evidence: the installed `skills/full-model/SKILL.md`, “standard Metal generator interface,” explicitly requires a concrete generator implementing `readiness_check.contract.Generator`. Installed `runtime/readiness_check/contract.py:69` defines `Generator(ABC)`; five abstract methods are `prefill_forward`, `decode_forward`, `prefill_logits`, `generate`, and `reset`. Its optional `prefill_logits_continuation` raises NotImplementedError by default. It provides a tokenizer class attribute, but no initialization or tensor computation. The autoport implements all five abstract methods and continuation (`generator.py:343,529,610,622,662,815`) and contains no `super()` calls. The teacher-forcing loader imports the generator factory by file convention; lack of an isinstance gate does not waive the explicit inheritance requirement.

Minimal repair recommendation:

- Add a small model-owned compatibility contract module under `tt/`; have `generator.py` import its `Generator` using an absolute autoport package import, consistent with the readiness file-loader convention.
- First attempt `from readiness_check.contract import Generator`. When successful, export that exact class, preserving actual installed-ABC inheritance and identity.
- Fall back only when `ModuleNotFoundError.name == "readiness_check"`. Re-raise other errors, including a missing `readiness_check.contract` in an installed package and missing transitive dependencies. Do not use `except ImportError` or catch-all fallbacks.
- For a genuinely absent package, provide a local `ABC` with the same five abstract method signatures/keyword contracts, tokenizer attribute, and optional continuation default. The fallback needs no runners, HF references, package installation, or skill path. Preserve applicable attribution if adapting the generic contract source.
- Do not use `object` as the fallback, erase abstract enforcement, inject a fake readiness module into sys.modules, or publish the installed skill package. An unconditional local duplicate would lose inheritance from the real installed ABC; conditional selection avoids that.

The literal identity of an absent class cannot be preserved on CI. This design preserves the real ABC where installed and its abstract interface on standalone CI. Keep the two environments in separate processes for tests; installing the readiness package after the generator was already imported cannot retroactively change its base class.

### 2. Default precision loads an excluded evidence file

Direct observations:

- `tt/precision.py:32-37` maps `None` to `doc/datatype_sweep/selected_precision_config.json` then reads it.
- `tt/model.py:24-25` defaults precision_config to None and calls this loader before HF/model allocation.
- `tt/generator_vllm.py:42` constructs that model without an override. Standalone generator construction has the same default (`tt/generator.py:38-45`).

Diagnosis: removing doc from the publication makes default model construction raise FileNotFoundError, even once the import blocker is fixed. An explicit policy dictionary bypasses the read, but the serving adapter does not supply one; changing only a standalone caller is insufficient.

The selected policy is **not BASELINE**. Its config_id is `head_bfp4`; LM-head weight policy is `bfp4` (bfloat4_b, LoFi), versus baseline `bfp8` (bfloat8_b, HiFi2). All other fields match the current baseline. The actual LM-head upload consumes that dtype at `model.py:77-78`; the compute configuration consumes its fidelity at line 103. Falling back to BASELINE changes tensor precision and computation, not just metadata.

Minimal repair recommendation: place the complete selected dictionary as a separate runtime-source constant in precision.py (or a small adjacent Python module), and resolve only None to that constant. Keep BASELINE as the existing schema/reference, retain explicit dictionary and string/Path overrides, the current validation, and deepcopy behavior. A Python constant avoids JSON package-data omission in source-only publication. Do not copy the evidence tree or fall back silently when an explicit path is missing. Treat the evidence JSON as provenance for this fixed policy, not as a runtime dependency.

Preserve every selected field: all four weight groups bfp4; all four compute fidelities LoFi; empty layer_exceptions; activation and KV cache bfloat8_b; prefill CCL bfloat8_b and decode CCL bfloat16; residual/logits/sampling/embedding/norm bfloat16; max_seq_len 262144; config_id head_bfp4. Snapshot the complete dictionary rather than deriving it from a mutable baseline.

## Exact-output preservation argument and limits

The proposed intervention changes only default-policy storage and base-class availability. Equal policy dictionaries flow into the same model construction, with identical dtype/fidelity/layout/cache choices. All implemented generator methods remain unchanged; neither base contributes executed model arithmetic, and all inherited public methods are overridden. Therefore no numeric or sampling change is implied by this packaging repair. Hardware token equality has not been measured here and must not be represented as proven by host-only import tests.

## Static checks performed and report adjudication

A host-only Python -B check loaded precision.py through runpy, then copied that single source into a temporary `model/tt` tree with no doc directory. Calling its current default reproduced FileNotFoundError at the expected missing selected-policy path. Passing the selected dictionary returned an exactly equal policy, and mutating a returned nested group left the input intact, confirming deepcopy behavior. No torch/TTNN model or weights were loaded.

An AST check compared installed abstract methods with the autoport class: all five are implemented, continuation is implemented, and there are zero super() calls. Re-reading the caller chain confirmed that serving does not already override precision and imports the generator eagerly. These checks support both headline causal chains; neither is merely a speculative numerical issue.

Recommended repair verification (not claimed as executed):

1. Compare the new no-argument policy with the full selected evidence dictionary; verify independent nested copies, explicit dictionary/path overrides, and invalid-policy errors.
2. In a fresh process with docs and skills runtime unavailable, import the published adapter with its normal Torch/TTNN/Transformers dependencies available, without constructing a model or opening a device. Check the resolved adapter is this autoport.
3. In a separate process with the real installed readiness runtime, assert issubclass(Gemma4Generator, readiness_check.contract.Generator) and that the concrete class is not abstract. Test that an incomplete local fallback subclass cannot instantiate.
4. Verify missing contract/transitive-dependency errors in an installed readiness package propagate rather than selecting fallback. Keep generator method bodies unchanged and inspect the source-only publication file list.

## Other dependency findings / limits

The audit covered all autoport tt/*.py import/file-access expressions and imported generic lightweight-module, CCL, sampling helpers and trace-allocation tracker. No additional required doc or developer-home input paths were found.

- `benchmark_phases.py:21` generates `/tmp/gemma4-phase-{digest}.sock`; this is an optional output/control socket, not an unpublished input. `generator_vllm.py:66-80` enables recording only with GEMMA4_BENCHMARK_PHASE_LOG. The recorder creates/appends the configured output path; local benchmark artifact paths are not hardcoded runtime prerequisites.
- `generator_vllm.py:57,71` emits realpath(__file__) as provenance, not a file dependency.
- HF config (`model.py:29`), tokenizer/generation config (`generator.py:55-56`), index and weight shards (`weights.py:10-23`) remain expected external assets, pinned to revision 842da3794eaa0b77d5f08bae87a17459d91ff475. Do not publish weights to repair packaging.
- CI requires a compatible Transformers version supplying `Gemma4TextRotaryEmbedding` (`model.py:12`), plus normal TTNN/Torch dependencies. Their actual CI availability was not tested; there is no inspected evidence to promote those to additional failures.

## Provenance

Installed contract and skill read from `/home/mvasiljevic/.codex-gemma4-v3/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/` (runtime/readiness_check/contract.py, skills/full-model/SKILL.md); also inspected readiness __init__.py and teacher-forcing loader and read the AutoDebug skill without launching it. Startup run_contract.json and agentic_evals/inputs.json were read.

- Selected JSON SHA256: af0969d87723dccb3a1199c01632ae720324f68c0287e8e327566a754be2d410
- Installed contract SHA256: 42417eab48a87b6e9c31dee15ef68bac5c67970e487a295f78ee7f968ef93358
- Root AUTODEBUG.md SHA256 before investigation: 639d3cd381ea7430886407234b6ae9e03aa47352a74b4d4c299ad459c3efcd7e
