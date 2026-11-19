# Sampling chat allowance investigation

Failing contract: the shared full sampling profile must pass its existing
bad-word and structured/plain answer assertions. The first run failed these
three nodes with content=None at original total caps 100 and 8. Its logs did
not retain full responses, so the exact original termination is unknown.

Independent diagnosis: `AUTODEBUG_reasoning_budget.md` found no concrete native
loader/cache/mask/protocol defect. Native structured completions exceeded the
original caps; eight completed paths give a conditional 1528-token extra
allowance. Native bad_words_3 sampled EOS without a final answer despite ample
room. The combined native control remains failed and cannot certify a global
budget. No runtime source change was justified by that EOS alone.

Discriminating experiment: keep every original shared prompt, seed, thinking
default and assertion; use the existing optional
`--tt-reasoning-token-budget=1528`. A model-local pytest observer records complete
SDK responses without changing arguments or object identity. Its fake-client
CPU validation passed, including restoration of the original method.

Full48 served focused result: **3 passed in 112.10 seconds**, on
`acceptance_tracking_v3`, C262144/B2/page32, selected precision, trace tracking,
async scheduling. All eleven responses stopped normally with non-null final
content. Seed0 badwords used112 completion tokens/106 reasoning tokens, fitting
the increased cap but exceeding original100. Structured choice used51 tokens,
regex88, JSON56 and plain82. This supports a cap issue on these observed served
paths; it does not reconstruct the original unrecorded failures or repair the
native seed3 trajectory. All original assertions passed. Root read all outputs;
structured answers are valid/on-topic, constrained greetings include awkward
spellings and a one-character answer. See `hf_reasoning_budget_outcome.md` for
separate native quality findings. Neither constrained set replaces the accepted
12-case matched qualitative suite.

Changed model-owned files: `tests/record_sampling_chat.py` (observation only)
and evidence/docs. No plugin/runtime/default/seed/assertion change was made for
this repair. The existing plugin option adds allowance to just these chat tests.
The full profile subsequently passed67 tests with75 existing skips in1109.31s;
run_vllm_server exited0. See `readiness_vllm/sampling_full_acceptance.json` and
`logs/sampling_full_budget1528_v1.log`. A passing served run cannot turn the native control
into a pass or establish a universal reasoning upper bound.

Artifacts: `readiness_vllm/sampling_chat_budget1528_v1/request_*.json`,
`logs/sampling_chat_budget1528_v1.log`, `observer_validation.json`, native
`readiness_vllm/hf_reasoning_budget_v1/manifest.json`, and independent report.
