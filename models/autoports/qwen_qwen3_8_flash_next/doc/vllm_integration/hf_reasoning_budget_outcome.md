# Native reasoning-budget outcome

The original 48-layer HF model completed all nine requests with normal EOS,
without hitting the 4096-token discovery cap or 7200-second per-case timeout.
The script exited 1: `budget_established=false`. Its stricter combined criterion
requires a reasoning-end marker and an answer satisfying the original assertion
for every case. No failure has been converted to a pass.

| Case | Reasoning tokens | Answer tokens including EOS | Result |
| --- | ---: | ---: | --- |
| choice_0 | 69 | 2 | blue; original assertion passes |
| regex_1 | 67 | 5 | LANE-1; original assertion passes |
| json_2 | 41 | 13 | status ok, lane 2; original assertion passes |
| plain_3 | 43 | 7 | Request 3 acknowledged; original assertion passes |
| bad_words_0 | 89 | 5 | Greet!; original ban assertion passes |
| bad_words_1 | 16 | 4 | Something.; original ban assertion passes |
| bad_words_2 | 29 | 4 | Hallo!; original ban assertion passes |
| bad_words_3 | absent | 0 | EOS at generated token 16, no final answer; fails |
| bad_words_4 | 763 | 6 | Pineapple!; original ban assertion passes |

Root read all raw outputs. The structured cases are coherent and on topic.
Bad-word masking applies to reasoning as well as the final answer. Cases 0/1
paraphrase the greeting prompt under the ban; case 1's final answer is not a
useful greeting. Case 2 invents a German-language request. Case 4 invents an
unrelated apple/hidden-reasoning exchange and a developer instruction, and
repeats its deliberation before answering off topic. These are quality findings,
not successful qualitative controls. No unreadable gibberish was observed.
The invented requests are absent from the actual recorded inputs.

A source audit found a fresh DynamicCache per generate call, no reused
past_key_values, and immutable embedding-row memoization only. The original
model's recurrent, convolution and PLE history live in that per-call cache.
EOS is legal before the reasoning-end marker. Increasing the cap cannot repair
case 3's sampled early EOS. HF and serving use different RNG algorithms, so its
seed 3 outcome does not establish the served seed 3 result.

Completed structured cases support an empirical extra allowance of 140 tokens
(twice 69 plus the reasoning-end marker). The largest completed bad-word case
would suggest 1528 (twice 763 plus the marker), with the original 100-token
answer allowance retained. These are starting allowances from completed paths,
not a combined-control pass or proof of unseen generations. The next discriminating experiment uses the existing optional allowance 1528
for the three failing shared test nodes, retaining every original assertion,
seed, prompt and thinking default. This is a diagnostic use of the completed
reference counts, not a claim that the combined native control passed. An
early-EOS failure on serving would remain a failure regardless of the cap.
The plugin README explicitly describes twice the largest completed count as
a starting allowance; no case hit the discovery or time ceiling. Independent
AutoDebug is still running, and any source finding will be checked before a
repair is accepted.

Exact requests, rendered IDs, raw generated IDs/text and per-case records:
`readiness_vllm/hf_reasoning_budget_v1/`. Execution log:
`doc/vllm_integration/logs/hf_reasoning_budget_v1.log`.
The existing thinking-off 12-case matched qualitative acceptance remains a
separate workload and is not replaced by these constrained default-thinking
controls.
