# Selected-profile qualitative assessment

Status: local selected sync suite and replay pass; independent stage review
and remote qualification remain separate gates. No observed optimization
regression on the matched greedy and seed-71 controls.

## Format and regression controls

The six prompts use `/v1/chat/completions`, one user message, the official
Gemma chat template with `add_generation_prompt=True`, and model/tokenizer
revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. Greedy temperature is zero;
sampled temperature/top-p are 0.7/0.9. Every request has a 256-token output cap.
The replay records tokenizer class, rendered text and IDs, server configuration,
usage, finish reasons, and three consecutive greedy responses per prompt.

All six selected sync official greedy outputs exactly match optimized async,
the original-path control, Stage 9, and Stage 10. The Stage 9/10 files are
byte-identical, not independent replications. The async and legacy replays
match all 18 greedy outputs and all six seed-71 sampled dictionaries. The final
sync replay matches both controls exactly too, including finish reasons, usage,
and all rendered prompt IDs. It freshly attests `GemmaTokenizer`; every greedy
prompt-usage count equals its rendered ID count. These are controlled regression checks,
not a claim that synthetic benchmark completions are useful prose.

The official suite's sampled requests are unseeded; their differing outputs
are retained, not treated as deterministic equivalence tests. The pinned
HF128 reference and selected TT128 controls provide shorter qualitative
references, not exact 256-token sampled controls. Prior standalone/serving
256-token controls reproduce the inherited story and thermodynamics greedy
wording in full.

## Observed limitations

The final sync suite has relevant, coherent answers without visible runaway
repetition or off-topic collapse. It also retains concrete defects:

- The story's greedy `tiny-brass-heart` and thermodynamics' `own-contained`
  wording are inherited. The final unseeded story has `no largeran a walnut`;
  final unseeded thermodynamics has `own-going`. These exact new draws were not
  reproduced. Prior sampled controls contain malformed compounds such as
  `pair-o-glasses` and `brass-bound-and-etched`, establishing the existing error
  class, not exact equality for every random draw.
- Thermodynamics overstates the second law in greedy output. The final sampled
  answer claims constant energy for a closed system, overlooking energy
  exchange; the prior sampled control already contains that factual error.
- The supervised/unsupervised comparison, story, thermodynamics, and Fibonacci
  answers are incomplete at the 256-token cap. This is a quality limitation,
  not hidden by a token-exact regression pass. The final sampled Fibonacci
  implementation computes the nth number sensibly but is also capped.
- The earlier async unseeded Fibonacci response mislabeled a materialized-list
  function as a generator. It remains in the historical draft, not silently
  replaced by the better final random draw.
- The seed-71 Fibonacci response describes an nth-number approach but returns
  a list; seed-71 thermodynamics retains `any-energy`/`of-disorder` wording and
  an overbroad entropy claim. These exact responses also occur in the legacy
  and async controls. Their errors are inherited, not excused by equality.

Haiku and French translation are readable and relevant. Matched greedy and
seeded parity can establish no observed optimization regression on these
controls; it cannot establish blanket factual accuracy, explain every typo,
or waive the model's inherited accuracy caveats.

Fresh greedy finish metadata confirms `length` at 256 tokens for prompts
1/2/3/5. Haiku stops at 19 tokens and translation at 100 tokens. Both full
sampling suites pass 72 tests with one expected cap-related skip; the selected
sync suite takes 1114.96 s. Machine control results are in
[final_qualitative_comparison.json](final_qualitative_comparison.json).

## Evidence

- [Selected suite](../../readiness_vllm/ttft_optimization/acceptance_sync_suite/readiness_vllm/vllm_qualitative_outputs.json)
- [Selected replay](../../readiness_vllm/ttft_optimization/acceptance_sync_chat_replay.json)
- [Legacy replay](../../readiness_vllm/ttft_optimization/legacy_chat_replay.json)
- [Prior 256-token controls](../../readiness_vllm/qualitative_256_controls.json)
- [Historical async assessment](draft_qualitative_verdict.md)

Canonical full sampling tests also exercise batches above the 32-slot serving
occupancy through queuing. Ordinary logprobs use the supported TP4 host
compatibility path, not newly claimed device logprob support. The expected
all-vocabulary logprobs-cap skip is reported separately from passes.
