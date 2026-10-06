# Draft qualitative verdict

Historical async-only draft, preserved as written. Its pending statements are
superseded by the [selected-profile assessment](qualitative_verdict.md) as the
final checks complete; its unseeded output observations remain evidence.

**All six current greedy completions exactly match both the optimized-vLLM
baseline and original stage9 outputs.** Their prompt texts/order also match.
The two historical output artifacts are byte-identical, so they are not two
independent replications. See [comparison and source hashes](qualitative_comparison.json)
and [current responses](../../readiness_vllm/ttft_optimization/acceptance_suite/readiness_vllm/vllm_qualitative_outputs.json).

This establishes no visible greedy-text regression on this saved six-prompt
suite. It is not final quality signoff or proof that warmed prefill replay was
exercised: the official suite alternates greedy/sampled requests, which can
invalidate the fast-path state. Three consecutive greedy requests and matched
seed71 sampled selected-versus-legacy controls are still pending.

## Formatting and controls

[Pinned format metadata](../../readiness_vllm/qualitative_prompt_format.json)
records `google/gemma-4-26B-A4B-it` revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, chat endpoint, one user message,
and tokenizer `apply_chat_template(add_generation_prompt=True)`. The rendered
template contains the BOS, user/model turn markers and model thought-channel
prefix. All six saved renderings and token-ID lists match the
[selected TT128 controls](../datatype_sweep/selected/qualitative_tt.json).
This was a saved-artifact comparison, not new tokenization. The current official
output file itself saves only prompt/mode/text, not revision, rendered IDs,
usage, or finish reason; the planned replay must supply that fresh attestation.

Official requests use greedy temperature0 or sampled temperature0.7/top_p0.9,
with max_tokens256. They do not record an effective sampled request seed. Every
current sampled response differs from the prior sampled output, which by itself
is neither a correctness failure nor evidence of a TTFT regression.

All current greedy texts extend or equal the selected TT128 outputs after
removing the latter's terminal `<turn|>` marker. The prior
[HF128 reference](../full_model/qualitative_hf.json) uses the pinned chat model
but differs in wording and has half the output cap; it is a qualitative
reference, not a text-exact 256-token control. In particular, it stops before
the problematic later story/thermodynamics wording, so its absence there is not
evidence that the new changes introduced it.

The prior [standalone256 controls](../../readiness_vllm/qualitative_256_controls.json)
cover story and thermodynamics and reproduce their entire greedy serving text,
including `tiny-brass-heart` and `In any own-contained system`. Both controls
generated 256 tokens. This supports inherited model/policy behavior rather than
a new serving-only corruption for those exact greedy phrases. It does not
excuse their quality or explain current unseeded sampled oddities.

## Visible quality observations

| Prompt | Observation |
|---|---|
| Haiku | Coherent relevant three-line poems with ordinary 5-7-5 syllable readings; greedy wording is inherited. |
| Supervised/unsupervised | Relevant teacher/labeled-example analogy. Both responses end during the unsupervised explanation, leaving the comparison incomplete. |
| Story | Coherent continuation with unusual compound wording: inherited `tiny-brass-heart`, sampled `brass-bound-and-glass`. Both end mid-sentence. |
| Thermodynamics | Greedy repeats the inherited malformed `own-contained` wording and overstates the second law as “will always increase.” Sampled says a closed system's energy remains constant, overlooking energy exchange. Neither reaches the third law within the saved output. |
| French translation | Readable, appropriate formal/informal variants; no visible truncation. |
| Fibonacci | Greedy iterative implementation is sensible, but the example/code fence is cut off. Sampled labels its function a memory-efficient generator while appending to and returning a list, with no `yield`: a substantive description/code mismatch requiring matched control. |

The abrupt endings are consistent with the 256-token cap and are inherited in
greedy outputs. The official artifact omits finish reasons, so this draft does
not claim a freshly verified `finish_reason=length` for every truncated row.
There is no visible runaway repetition or off-topic collapse in the inspected
responses, but incomplete answers and the concrete errors above remain quality
limitations. Seeded comparisons and replay evidence are required before the
draft can become an accepted stage verdict.
