# AutoFix: standalone source packaging

Status: repaired and host-confirmed; no hardware or live CI validation claimed. Used device hours: 0 of the coordinator's 0.1-hour box. Change: kept; packaging problem: solved under the tested import/configuration contract. The coordinator owns final acceptance and publication.

The independent `AUTODEBUG.md` identified two deployment blockers. Verified against current code: default `load_precision(None)` read an excluded documentation file, while `generator.py` unconditionally imported the optional readiness runtime. The selected head policy differs from BASELINE, so replacing it with BASELINE would change numerical behavior.

## Repair

- `tt/selected_precision_config.json`: exact runtime copy of the selected policy; required source configuration included in publication alongside Python files.
- `tt/precision.py`: default path points to that adjacent configuration. Explicit policies/paths and validation remain unchanged.
- `tt/generator_contract.py`: uses the exact installed readiness ABC when available. Only a ModuleNotFoundError naming the missing root `readiness_check` selects the local ABC; missing submodules/transitive dependencies propagate. Fallback preserves all five abstract signatures, tokenizer attribute, and continuation NotImplementedError default.
- `tt/generator.py`: imports that compatibility contract. The complete Gemma4Generator class AST is identical to execution HEAD b030d0b10a88ec05933bf95bc106b830f5166b3b.
- `tests/check_source_packaging.py`: focused host regression checks with TTNN/device-dependent imports stubbed, never constructed or opened.

The coordinator selected an adjacent JSON instead of the diagnosis's preferred Python constant because publication explicitly includes runtime configuration. This avoids copying documentation and preserves the selected artifact byte for byte.

## Confirmation

`READINESS_CONTRACT_PATH=/home/mvasiljevic/.codex-gemma4-v3/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/runtime/readiness_check/contract.py python models/autoports/google_gemma_4_31b_it/tests/check_source_packaging.py -v`

All 4 checks passed: doc-free copied runtime loads the exact policy, nested copies/explicit overrides retain behavior, copied generator and adapter import with readiness unavailable, incomplete fallback cannot instantiate, fallback continuation raises, installed ABC identity/inheritance and evaluated signatures match, concrete generator has no abstract methods, installed-package failures propagate. Standalone and installed-ABC individual tests were additionally rerun in separate fresh Python processes and passed. Device-dependent module stubs mean this proves import wiring, not compatibility of CI's actual Torch/TTNN/Transformers installations.

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --confcutdir=models/autoports/google_gemma_4_31b_it/tests models/autoports/google_gemma_4_31b_it/tests/test_precision_policy.py -q`

7 passed. Parent device fixtures and third-party plugin autoload disabled; one benign warning for the repo timeout option whose plugin was disabled.

`python models/autoports/google_gemma_4_31b_it/tests/check_serving_host_contract.py`

Passed existing host-only control-flow checks (no TTNN import/device use), including parameter padding, RNG state, stale inputs, scheduler blocks, and token-range validation.

A host AST/bytes check compared `git show HEAD:models/autoports/google_gemma_4_31b_it/tt/generator.py` with the worktree and compared both selected JSON files. Complete class AST equal; selected configuration bytes equal, SHA256 `af0969d87723dccb3a1199c01632ae720324f68c0287e8e327566a754be2d410`. All numerical implementation bodies and selected construction values are unchanged. No token equality, speed improvement, or hardware result was measured.

## Remaining boundary

Publication must include `tt/selected_precision_config.json` and `tt/generator_contract.py`. Actual deployment dependency compatibility and live loading remain the coordinator's CI/preflight checks. Installed skills and plugin runtime were read-only; root AUTOFIX.md and unrelated worktree changes were preserved. No reset, device command, push, or CI dispatch occurred.
