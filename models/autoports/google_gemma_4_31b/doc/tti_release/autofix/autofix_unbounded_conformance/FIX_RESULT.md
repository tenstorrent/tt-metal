# AutoFix Result: bounded determinism conformance samples

## Starting evidence

- Source diagnosis: `AUTODEBUG.md` in this directory.
- Original symptom: the vLLM parameter suite left a short reproducibility request decoding toward the 113,280-token context remainder because two payload constructors omitted `max_tokens`; `timeout=None` correctly allowed that request to keep progressing.
- Scope of this repair was host-only. Per the parent task, no live server, TT device, reset, or other hardware action was run here.

## Hypothesis experiment

- Hypothesis: adding an explicit 32-token observation window only to `test_seed_reproducibility` and `test_determinism_parameters` prevents the accidental context-remainder default while preserving the tests' actual assertion (two non-empty outputs are equal).
- Prediction: both reproducibility calls and all six determinism-parameter calls carry `max_tokens=32` and `timeout=None`; neighboring tests retain their independent limits and timeout behavior.
- Focused experiment: invoke the conformance functions through a recording client and assert the complete payload and client kwargs.
- Result: verified. The focused regression has six passing cases and covers the two bounded payloads plus the neighboring 50-token non-uniform seed, 100-token logprobs, 1024-token penalty, and short 32-token/default-timeout behavior.

## Fix

- Added `"max_tokens": 32` to only these production payload constructors:
  - `test_seed_reproducibility`
  - `test_determinism_parameters`
- Retained `timeout=None` on every affected call.
- Renamed the prior regressions that claimed no cap, changed them to require 32 tokens, and added explicit penalty coverage proving the intentional 1024-token sample remains unchanged.
- Did not change the release model spec, server context, prompts, evals, benchmarks, global request behavior, or any other conformance token budget.

## Verification

- Focused regression:
  - `.workflow_venvs/.venv_v2_run_script/bin/python -m pytest -c pyproject.toml --rootdir . --confcutdir . tt-inference-server-v2/tests/llm_module/test_vllm_chat_completions_timeouts.py -q`
  - Result: `6 passed`.
- Broader host-only tests:
  - `.workflow_venvs/.venv_v2_run_script/bin/python -m pytest -c pyproject.toml --rootdir . --confcutdir . tt-inference-server-v2/tests/llm_module -q`
  - Result: `63 passed`.
- Production suite collection under the isolated TTI pytest boundary:
  - `.workflow_venvs/.venv_v2_run_script/bin/python -m pytest -c pyproject.toml --rootdir . --confcutdir . tt-inference-server-v2/llm_module/test_vllm_chat_completions.py --collect-only -q`
  - Result: `21 tests collected`.
- Focused live functional regression against the external autoport server:
  - `test_seed_reproducibility` plus all three `test_determinism_parameters`
    cases, using the real OpenAI-compatible endpoint.
  - Result: `4 passed, 17 deselected in 91.88s`.
  - All eight bounded requests completed; no timeout or unbounded context-remainder
    generation occurred, and the equality assertions executed successfully.
- Formatting and diff:
  - `black --check tt-inference-server-v2/tests/llm_module/test_vllm_chat_completions_timeouts.py`: pass.
  - `black --check --line-ranges 106-116 --line-ranges 224-236 tt-inference-server-v2/llm_module/test_vllm_chat_completions.py`: pass for the changed production regions. A whole-file Black 26 run would reformat pre-existing assert style, so no unrelated formatter churn was retained.
  - `git diff --check`: pass.
  - Reviewed production diff: limited to the two `max_tokens=32` additions.

## Final status

- Repair: fixed with focused, broader, and live-server evidence.
- Remaining required evidence: the parent must run the complete release gate against the autoport server.
- Changes are intentionally uncommitted.
