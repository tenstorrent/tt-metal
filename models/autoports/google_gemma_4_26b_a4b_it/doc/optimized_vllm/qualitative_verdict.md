# Qualitative serving regression check

Read all six greedy and six sampled outputs from the candidate full30-layer
server. All six greedy outputs match the preceding validated serving control
exactly, with identical prompts and chat mode. qualitative_comparison.json
records the comparison; inherited/vllm_qualitative_outputs.json is the control.
Pinned HF/selected-policy controls and rendered prompt metadata are preserved
by ../vllm_integration/qualitative_verdict.md and inherited/qualitative_prompt_format.json.
Requests use /v1/chat/completions, not raw instruction completions.

The haiku is coherent and on topic. The learning explanation correctly contrasts
teacher/labeled learning with discovery; long answers truncate at256 tokens.
The Elara clockwork-heart story remains coherent. Greedy thermodynamics retains
the controlled malformed phrase “own-contained”; candidate sampled text uses
“isolated system” and a coherent entropy explanation. French translations match
the requested greeting and distinguish register. Both Fibonacci answers contain
reasonable iterative Python functions. No new doubled tokens, mechanical loops,
wrong-language drift, gibberish or request contamination was observed.

This is a serving-regression check, not a broad accuracy certification. Existing
scientific wording and output-length limitations remain; sampled outputs are
not expected to exactly match unseeded controls. The inherited device sampler
caps effective stochastic top-k at32; greedy semantics are unchanged.
