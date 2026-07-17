# AutoDebug Report: unbounded vLLM determinism conformance requests

## Scope and starting evidence

Inspection only; no source files were edited and no hardware experiment was run.

- Failing run: `.exp_run/tti-release/gemma4-31b-20260716/tti_release_final5.log`, whose spec suite starts at line 1130 at 20:19:52 UTC.
- Live-server evidence: `models/autoports/google_gemma_4_31b/readiness_vllm/server.log` lines 10462 and 10531-10559 show one request continuously decoding at about 1.4-1.6 tokens/s while KV use grows from 0.1% to 1.0%; line 10561 shows it was terminated at 20:36:53. This is forward progress, not a TT device hang.
- TTI revision inspected: `507c74673d07f86aa9955a226fc8f56f29aa696c` (`Allow long vLLM conformance requests`).
- vLLM revision inspected: `44b7853d448f3f8c5db7ed068a4f82ebfcd1065d`.

## Headline finding: two functional determinism payloads accidentally request the context remainder

Confidence: high.

The exact affected production call sites are:

1. `tt-inference-server-v2/llm_module/test_vllm_chat_completions.py:108-111`, `test_seed_reproducibility`: the payload has `messages`, `seed`, and `temperature`, but no `max_tokens`; it is submitted twice with `timeout=None`.
2. The same file at lines 222-229, `test_determinism_parameters`: each of the `temperature=0`, `top_k=1`, and `top_p=0.01` payloads omits `max_tokens`; each is submitted twice with `timeout=None`.

This is not merely a generous timeout. vLLM's chat request schema defaults omitted `max_tokens` to `None` (`vllm/entrypoints/openai/chat_completion/protocol.py:158-163`). Chat serving passes that value to `get_max_tokens` (`chat_completion/serving.py:386-401`). `get_max_tokens` defines the fallback as `max_model_len - input_length` (`vllm/entrypoints/utils.py:175-193`). With this server's preserved `max_model_len=113280` and a short prompt, each affected API request can therefore request nearly the entire 113,280-token context remainder. The final5 log behavior is the predicted consequence.

Commit `507c74673` made these requests `timeout=None`, but deliberately left the payload unchanged. Its regression tests currently encode the bad behavior with `assert "max_tokens" not in expected_payload` at `tt-inference-server-v2/tests/llm_module/test_vllm_chat_completions_timeouts.py:45` and `:114`. Removing the client timeout exposed the pre-existing unbounded-request bug.

No other payload in this module has the same defect. `test_non_uniform_seeding` already uses `max_tokens=50`, `test_n` uses 32, `test_stop` uses 20, `test_logprobs` uses 100, and penalty tests use an intentional 1024-token sample.

## Correct intervention

Set `"max_tokens": 50` in both functional determinism payload constructors:

- `test_seed_reproducibility`, line 108.
- `test_determinism_parameters`, line 222, before the optional high-temperature field is added.

Fifty is the evidence-backed value rather than an arbitrary release-only alignment:

- The adjacent and stronger `test_non_uniform_seeding` test already uses 50 tokens to prove seed behavior across 32 concurrent requests.
- These two tests assert only that two non-empty outputs are identical. They do not assert EOS, response length, long-generation quality, or context capacity. A fixed finite trajectory is the intended observation window; generating tens of thousands of additional tokens does not add a new assertion.
- The prompt explicitly asks for a concise capital-of-France answer, so 50 completion tokens are ample for a meaningful non-empty sample while keeping all eight affected calls (two seed calls plus two calls for each of three deterministic parameters) bounded.
- It avoids introducing a model-, device-, page-, tile-, trace-, or benchmark-aligned value. The same request semantics apply to every backend.

The exact regression-test updates should be:

- Rename `test_seed_reproducibility_disables_timeout_without_changing_payload` to describe a bounded payload, add `"max_tokens": 50` to `expected_payload`, and remove the negative assertion at line 45.
- Rename `test_determinism_parameters_disable_timeout_without_adding_token_cap` similarly, add `"max_tokens": 50` to every `expected_payload`, and remove the negative assertion at line 114.
- Retain `timeout=None`. A finite completion budget and an unlimited wall-clock client timeout are independent: the former defines the functional sample; the latter avoids falsely failing slow but progressing hardware.

## Why this preserves the release context contract

This change does not alter `doc/context_contract.json`, the release spec, server `--max-model-len 113280`, tokenizer behavior, prompt length, prefill logic, request admission, or any benchmark/eval length. It does not truncate or reject an incoming prompt. It defines the intended completion budget on two small functional API requests, exactly as the other parameter-conformance cases already do.

It also cannot conceal the non-divisible/context bugs forbidden by the stage contract: those are exercised independently by the non-aligned smoke, eval requests, and the 17-point benchmark sweep (including the successful 65,535-token input in final5). These determinism tests are not context-capacity tests. The new prompt-plus-50 request is mathematically valid under 113,280, and the server remains obligated to accept valid requests up to that contract.

The bounded test has a narrower and clearer claim: the specified seed/sampling parameter yields identical first 50 completion tokens. As with any finite functional test, it cannot prove infinite-horizon determinism, but omission did not provide such a proof either: model EOS could end an uncapped request at any arbitrary point. Long-context and long-output behavior belong to explicit eval/benchmark rows, not an accidental API default.

## Validation plan for the repair

1. Focused payload test (no server):

   `.workflow_venvs/.venv_v2_run_script/bin/python -m pytest -c pyproject.toml --rootdir . --confcutdir . tt-inference-server-v2/tests/llm_module/test_vllm_chat_completions_timeouts.py -q`

   Require both calls from `test_seed_reproducibility` and all six calls from the three `test_determinism_parameters` cases to contain `max_tokens=50` and `timeout=None`.

2. Focused live functional test against the existing autoport server:

   Run `tt-inference-server-v2/llm_module/test_vllm_chat_completions.py` with the same endpoint/model/output fixtures as final5 and `-k 'test_seed_reproducibility or test_determinism_parameters'`. Require 4 pytest cases / 8 HTTP 200 responses, non-empty equal output pairs, and no request continuing past 50 completion tokens.

3. Original gate: rerun the complete vLLM parameter suite, then the release workflow/report. Require all parameter cases to pass and preserve the final5 eval/17-point benchmark/context metadata.

4. Static review: verify no release model spec, server max context, eval limit, benchmark ISL/OSL, or prompt content changed in the repair diff.

## Other observations

- `test_penalties` intentionally uses 1024 tokens and may be slow, but it is bounded and uses length/diversity statistics over the generated sample; it is not the root cause of the final5 unbounded request.
- Changing only a Requests timeout, adding a global server output cap, lowering `max_model_len`, or shortening the release benchmark would intervene at the wrong boundary and would violate or weaken the release contract.

## Verdict

Root cause verified by direct source and runtime evidence. The smallest correct fix is `max_tokens=50` at the two payload-construction sites plus the corresponding payload regression expectations. No model/generator/device failure is implicated by this symptom.
