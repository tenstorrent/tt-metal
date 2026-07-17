# AutoDebug: v2 ignores external runtime model specs

## Scope

- Checkout: `tt-inference-server` detached at `v0.18.0`, SHA `d5913e816ac5dc33d86c1f3f5960348bc3fa4e2e`.
- Target spec: `.exp_run/tti-release/gemma4-31b-20260716/autoport_smoke_spec.json`.
- Required implementation: `models/autoports/google_gemma_4_31b` for `google/gemma-4-31B` on `p150x4`.
- Investigation was inspection-only. No source, server, or hardware state was changed.
- The checkout already had unrelated live changes in `evals/eval_config.py` and v2 test configuration; they were treated as existing stage work.

## Evidence-ranked findings

### P1 / verified: v2 re-applies built-in-catalog model validation at two gates, defeating the external runtime spec

This is the complete causal chain for the target failure.

1. Top-level v1 correctly defines `--runtime-model-spec-json` as authoritative: `run.py:730-741` loads `ModelSpec.from_json(...)` instead of `get_runtime_model_spec(...)`.
2. The v2 bridge then forwards both the resolved custom model name and the generated combined runtime-spec path: `workflows/v2_bridge.py:200-214` uses `model_spec.model_name` and `--runtime-model-spec-json`.
3. The target custom spec's resolved name is `gemma-4-31B`, which is absent from v2's built-in model choices (only stock `gemma-4-31B-it` exists). `tt-inference-server-v2/run.py:107-122` therefore rejects the real bridge command at argparse before reading the supplied runtime spec.
4. If that first gate is bypassed (as in the focused repro, or by using the stock alias), `tt-inference-server-v2/workflow_module/command_factory.py:97-106` still unconditionally calls `get_runtime_model_spec(model=args.model, device=args.device)`. It loads only the built-in catalog and never calls `ModelSpec.from_json`, producing the observed `ValueError` because no built-in Gemma 4 spec supports `p150x4`.

Direct evidence:

- Exact focused repro exits 1 at `command_factory.py:100` with `ValueError: Model:=gemma-4-31B does not support device:=p150x4 in the 'prod' catalog`.
- A full direct v2 invocation with `--model gemma-4-31B` exits 2 in argparse as an invalid model choice.
- Using the built-in alias `--model gemma-4-31B-it` gets past argparse but exits 1 at the same `_build_context` catalog lookup for `p150x4`.
- Both existing deserializers successfully load the same flat custom JSON: `ModelSpec.from_json` yields model `gemma-4-31B`, device `P150X4`, engine `vLLM`, and code path `models/autoports/google_gemma_4_31b`; `RuntimeConfig.from_json` yields external API mode on port 8000. These loaders support flat legacy and combined runtime-spec formats (`workflows/model_spec.py:677-701`, `workflows/runtime_config.py:228-249`).
- A no-edit in-memory control replacing the catalog lookup with `ModelSpec.from_json` made `_build_context` complete and printed:
  `CONTROL_OK gemma-4-31B models/autoports/google_gemma_4_31b P150X4 False False http://127.0.0.1:8000`.
- With that custom model object, downstream config builders also work without catalog membership: the smoke benchmark resolves exactly `(ISL=8, OSL=8, concurrency=1, prompts=1)`, and the current stage eval config resolves `meta_ifeval`.

Smallest correct intervention boundary:

- In v2, load `ModelSpec.from_json(args.runtime_model_spec_json)` whenever the path is present; use `get_runtime_model_spec(...)` only when it is absent. A shared helper should be used by `_build_context` and auth-engine resolution.
- Permit a non-catalog `--model` when `--runtime-model-spec-json` is present, while retaining built-in validation when no runtime spec is supplied. The smallest target-only alternative is to have the bridge pass the outer catalog alias, but that leaves direct external-spec v2 broken and risks selecting the wrong eval identity; it is not a complete fix.
- Validate that CLI device/model and loaded-spec identity are coherent where appropriate, but do not replace the custom spec with a stock spec.

### P1 / verified adjacent identity bug: eval selection uses the outer model alias instead of the loaded spec

`command_factory.py:113-114` calls `_resolve_eval_config(args.model)`, even though every downstream LLM eval selector uses `ctx.model_spec.model_name`. For this target, v1 must accept the stock CLI alias `gemma-4-31B-it`, while the custom runtime spec intentionally resolves the base checkpoint `gemma-4-31B`. If a repair merely forwards/retains the stock alias to satisfy argparse, `_build_context` would select the `-it` eval configuration even after correctly loading the base autoport spec. That would evaluate the wrong prompt contract and accuracy rows.

The repair should resolve eval metadata with `model_spec.model_name`, not `args.model`. The current stage edit registers `EVAL_CONFIGS["gemma-4-31B"]`, so the correct custom identity has an explicit base-model configuration.

### P2 / verified adjacent identity bug: auth-engine selection also ignores the runtime spec

`command_factory.py:_resolve_auth_token` (around lines 293-307) independently calls `get_runtime_model_spec(model=args.model, device=args.device)` to decide whether to send a literal Forge/media key or a vLLM JWT. For this target that lookup fails and the defensive fallback chooses the JWT branch. Because the smoke is external vLLM with `--no-auth`, the resulting empty token is harmless here. For any custom Forge/media spec absent from the catalog, however, the same bug chooses JWT semantics and can cause the documented 401 behavior.

The auth helper should inspect the same already-loaded external `ModelSpec` (or the same shared spec-loader helper) used by `_build_context`. Add coverage for custom vLLM and custom Forge/media specs so the engine cannot silently come from an unrelated catalog entry or fallback.

## Fix ancestry

- Commit `6e396b439` (`Support external runtime specs in release workflows (#4345)`, 2026-06-26) is an ancestor of this tag. It fixed v1 validation and benchmark construction so a resolved external spec does not need membership in `MODEL_SPECS`.
- The v2 command factory's unconditional catalog lookup dates to its initial implementation (`8c53e0defb`, 2026-05-29) and was not updated by #4345.
- Commit `695d263d5` (`route LLM evals/benchmarks through tt-inference-server-v2 (#4490)`, 2026-07-08) made external LLM benchmark/eval/release runs reach this older v2 code. Thus #4345 is present, but the later routing exposed an unported v2 assumption.
- The v2 unit-test module explicitly says `_build_context` is out of scope (`tt-inference-server-v2/tests/workflow_module/test_command_factory.py:5-10`). Existing tests cover runtime-config loading and mocked auth engines, but not external ModelSpec loading, custom parser identity, or a full bridge-to-context path. That explains why the regression passed tests.

## Focused verify/refute experiments for AutoFix

1. Add a pure unit test with a temporary custom combined runtime JSON whose model name, device, implementation, and engine are absent from `MODEL_SPECS`. Assert `_build_context` returns that exact `model_id`, `device_type`, `inference_engine`, and `impl.code_path`. This directly verifies/refutes the proposed loader boundary without running a server.
2. Add parser tests:
   - custom non-catalog model plus `--runtime-model-spec-json` parses;
   - the same non-catalog model without a runtime spec is rejected;
   - malformed/missing runtime JSON fails explicitly rather than falling back to stock.
3. Test both supported document shapes: the current combined `{runtime_model_spec, runtime_config}` format produced by top-level v1 and the flat `cli_args` legacy format used by the supplied temporary spec.
4. Use a custom spec whose outer CLI alias differs from `ModelSpec.model_name`; assert `ctx.all_params` comes from the loaded model name. For this target, assert base `gemma-4-31B` selects the base eval config and never `gemma-4-31B-it`.
5. Parameterize auth tests with external vLLM and Forge/media specs absent from the catalog. Assert vLLM follows JWT/no-auth semantics and Forge/media uses the literal API-key branch.
6. Run the exact focused repro. Expected output must include `gemma-4-31B models/autoports/google_gemma_4_31b` and no catalog lookup error.
7. Run a no-server construction-level bridge test (mock subprocess execution): top-level external spec -> generated combined runtime JSON -> v2 argv -> `parse_args` -> `CommandFactory.build`. Assert the resulting context still names the autoport path. This covers the argparse gate that the focused repro bypasses.
8. Run the existing focused command-factory/parser suite, then the tiny real external-server benchmark smoke. The smoke run spec/report must retain `docker_server=false`, port 8000, context 113280, and `models/autoports/google_gemma_4_31b`.

## Other observations

- Do not work around the failure by adding a stock catalog row or changing the custom implementation path. That would weaken the proof that TTI evaluated the generated autoport.
- Do not rename the custom base model to `gemma-4-31B-it` merely to satisfy argparse. Besides changing model identity, it triggers the wrong eval/prompt contract unless every downstream selector is also corrected.
- The target JSON's embedded legacy `cli_args.model` is the outer stock alias while its authoritative `ModelSpec.model_name` is the base model. This is why code must separate CLI admission from resolved model identity. The generated combined runtime spec produced by top-level v1 remains the best artifact to pass to v2/reporting.

## Conclusion

The failure is a software harness regression, not infrastructure or model behavior. External-spec precedence was fixed in v1 but not propagated into the v2 parser, context builder, eval identity, or auth-engine selection. The primary hypothesis is verified by the exact repro, successful native deserialization, and a focused in-memory control. Repair both v2 catalog gates and make the loaded ModelSpec the single identity source before rerunning the TTI smoke.
