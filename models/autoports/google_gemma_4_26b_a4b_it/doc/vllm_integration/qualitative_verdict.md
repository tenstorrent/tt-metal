# Shared serving qualitative review

The six greedy and six sampled full30-layer outputs in
`../../readiness_vllm/vllm_qualitative_outputs.json` were read, alongside the
pinned-revision HF controls and the selected datatype-sweep TT controls.
The model is instruct/chat. Requests use `/v1/chat/completions`; metadata and
rendered token IDs are in `../../readiness_vllm/qualitative_prompt_format.json`.
All six rendered prompts/token sequences match the prior controls exactly.

All six greedy serving strings preserve the selected-policy128-token control
prefix (short outputs omit the expected turn terminator). See
`../../readiness_vllm/qualitative_control_comparison.json`. Serving allows256
output tokens here, while both HF and earlier TT controls used128.

| Prompt | Greedy and sampled observations |
| --- | --- |
| shared_0 | Coherent three-line machine-learning haiku, “Data flows through nodes”; the sampled final line is “Learning from the code,” still coherent and on topic. |
| shared_1 | Greedy labeled-fruit teacher analogy reaches unlabeled grouping. Sampled labeled-data explanation reaches a spam-filter example; the 256-token cap cuts it before unsupervised learning. Both are coherent; the sampled answer is incomplete. |
| shared_2 | Greedy Elara/clockwork-heart story matches the selected control; final sampled Elian/lost-things compass story remains a coherent continuation. Some creative word choices are awkward, without a repetition loop or loss of topic. |
| shared_3 | Explains conservation of energy and entropy; mentions the zeroth-law convention. Long response is cut at256 tokens before covering every law. |
| shared_4 | Formal/informal French greetings answer the requested translation. English framing matches HF/control behavior and is not wrong-language drift. |
| shared_5 | Greedy produces an iterative list with correct updates; sampled computes the nth value and correctly gives fibonacci_iterative(10)=55. Greedy example and sampled second implementation are truncated at256 tokens. This review does not claim the entire requested explanation completed. |

Verdict: coherent, on-topic shared-suite smoke. No mechanical token/subword
repetition, collapse, gibberish, unexpected language switch, prompt echo or
visible cross-request text contamination was observed. The degeneracy checker
exited0 with no findings (`../../readiness_vllm/degeneracy.json`). Output caps are
explicit test limits, not a reduced served context. Reduced-layer diagnostic
responses are deliberately excluded from this quality verdict.


## Extended controls requested by independent review

`qualitative_256_controls.json` now contains full30-layer standalone selected-policy
controls at256 tokens for shared_2/shared_3, using exactly the saved pinned
chat-template prompts. Both greedy visible strings match serving exactly for
all256 tokens. The story's `tiny-brass-heart` and thermodynamics'
`In any own-contained system` therefore reproduce outside vLLM. The latter is
malformed wording; this review does not certify scientific correctness. It is
an observed selected-policy output limitation, not unexplained serving drift.

New sampled controls (temperature0.7,top_p0.9,effective top_k32,seed42) were read
in full: a coherent Elian/projected-map story and an on-topic conservation/entropy
explanation, both truncated at256. The story includes `brass-bound-and-glass`,
showing similar over-hyphenation in standalone generation. These are new draws;
the original unseeded shared responses have no saved sampled token IDs or seed,
so this does not claim exact reproduction of `pair-o-glasses` or
`brass-bound-and-etched`. Exact full-vocabulary standalone/adapter numerical
checks and live logprob checks independently support the model-path contract.
The final restarted-server suite was read in full. All six greedy prefixes again match the selected-policy control, and both extended greedy strings still match exactly; `qualitative_control_comparison.json` records the final artifact hash. The final sampled story again uses `brass-bound-and-etched`; sampled thermodynamics repeats `own-contained` and adds the awkward `measure of-disorder`. The same malformed greedy phrase and standalone over-hyphenation controls, together with exact numerical comparisons, support classifying these as inherited output limitations. There are no mechanical loops, gibberish, wrong-language drift, or visible request contamination. This is a serving-regression smoke verdict, not a claim of scientific or broad task accuracy. The final degeneracy run exited0 with no findings.

All four controls used trace allocation tracking with program-cache checking
retained and completed without an unsafe-survivor error. They are functional
controls, not performance measurements.
