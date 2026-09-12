# Full-model qualitative review — pass for the recorded suite

Date: 2026-09-12. Independent review under the qualitative-check skill. This reviewer read all twelve original outputs and all eight extended outputs, compared saved tokens, and ran the shared degeneration checker. No Torch/TTNN imports, model execution, hardware access, or implementation changes were performed by the reviewer. The coordinator reports the extended hardware process/launcher exited 0 and closed its device.

**The recorded six-prompt qualitative suite passes with the matched 1024-token extensions for p0, p2, p3 and p5, plus the completed 256-token p1/p4 results.** All six selected TT answers reach EOS and fulfill their requests coherently. No materially worse TT final answer, wrong-language answer, cross-request leakage, gibberish or mechanical repetition is visible. HF's p2 story remains coherent but capped at 1024; TT supplies a complete story at 531 tokens.

The original 256-token run remains insufficient for a full-suite pass. Its review is preserved unchanged as `qualitative_review_256.md`. In particular, the extended TT p0 changes its earlier prefix: the new completed haiku does **not** prove that the original truncated haiku branch would eventually terminate. This limitation is recorded below, rather than treating the rerun as an exact continuation.

## Prompt and control integrity

The artifacts identify `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, Qwen2Tokenizer, chat mode with the checkpoint chat template, and greedy generation. The shared source is `models/common/readiness_check/vllm_prompts.txt`, SHA256 `2ad452c15d8442d3fe641eb968bbd84e3bfe9acb01ddc540f451983c0da3cbbd`; the source file matches the recorded hash.

For every original and extended pair, host comparisons verify identical prompt strings, rendered chat strings and exact prompt token IDs. Every rendered prompt includes the same xhigh reasoning system message and opens an assistant `<think>` section. The HF extension runs prompts `[0,2,3,5]` with a 1024-token cap; the TT extension uses those same rendered prompts and cap. HF uses a left-padded batch of four in the extension, versus six originally; TT submits each prompt separately. All HF saved prefixes are nevertheless identical across the two runs.

Completed raw text retains `</think>` and its final `<|im_end|>` because the runners decode the recorded token arrays with special tokens present. Both controls exhibit these expected markers. This is not evidence of an unexpected UI/serving leak. The reasoning preambles sometimes repeat the current user's request on both HF and TT; no other request's content appears. The final French answer is French even though the matched reasoning preambles are English.

Both artifacts retain text/tokens through the first configured EOS. TT still executes the requested fixed generation length internally, so 1024-token performance counters do not mean a saved 418-token answer lacked EOS. This review assesses the saved first-EOS completion, and makes no performance claim from the qualitative runs.

## Token boundaries and comparisons

All indices below are **zero-based generated-token indices**. Counts include the terminating EOS when present. “Divergence” is the first TT/HF token mismatch in that prompt's selected pair; early free-running divergence is informational, not itself a quality failure.

| Prompt / selected cap | TT / HF token counts | TT / HF `</think>` index | TT / HF EOS index | First TT/HF divergence |
| --- | --- | --- | --- | --- |
| p0 haiku / 1024 | 418 / 284 | 400 / 267 | 417 / 283 | 16 |
| p1 learning types / 256 | 148 / 133 | 31 / 31 | 147 / 132 | 44 |
| p2 story / 1024 | 531 / 1024 | 185 / 61 | 530 / absent, capped | 34 |
| p3 thermodynamics / 1024 | 433 / 411 | 187 / 184 | 432 / 410 | 14 |
| p4 French / 256 | 145 / 113 | 134 / 102 | 144 / 112 | 3 |
| p5 Fibonacci / 1024 | 337 / 237 | 163 / 81 | 336 / 236 | 17 |

`qualitative_prefix_comparison.json` records the extended comparisons and artifact hashes. HF preserves all 256 prior tokens for p0/p2/p3, and its entire 237-token p5 completion. TT preserves all 256 prior tokens for p2/p3/p5. TT p0 first changes at generated token 32: old token 16018 versus new token 10380, corresponding to the old “Could be:” versus new “Maybe:” wording. Its later rejected-line sequence also differs.

## Actual answer assessment

- **p0, machine-learning haiku:** TT delivers “Data flows in light / Models learn from hidden signs / Clarity forms now.” With Data=2, Models=2, hidden=2 and Clarity=3 syllables, the lines have 5/7/5 syllables. They relate to data, model learning and emerging understanding. HF delivers the valid alternative “Data flows in light / models learn from patterns deep / answers emerge now.” TT considers several four-syllable alternatives in its reasoning, then resolves them and terminates at 418 tokens. The observed extended run is not stuck; it is longer than HF and is not the exact old branch.
- **p1, supervised/unsupervised learning:** Both complete concise explanations of labeled examples versus discovering patterns/groups without supplied answers. TT uses cat recognition and photo grouping; HF uses cats/dogs and customer groups. The examples support the distinction, and neither answer contains mechanical repetition or a language mismatch.
- **p2, inventor story:** TT sustains one unnamed male inventor, the brass key, a door labeled “Memory,” and a coherent transition from opening physical doors to restoring shared memories. It ends with the inventor helping the kingdom open “the door to hope.” The magical memory-lantern machine connects his role as inventor to the ending. HF follows a different coherent story about Elara, her grandmother's music box and the Workshop of Forgotten Ideas, but stops at “The machine” at its 1024-token cap. TT satisfies the completion request; no HF completion is claimed. Repeating the supplied story fragment at the start is appropriate to this request, and neither story displays a repeated-loop failure.
- **p3, thermodynamics:** Both deliver the first, second and third laws plus a zeroth-law note. TT defines `Delta U = Q - W` with Q as heat added and W as work done by the system, states nondecreasing entropy for an isolated system, and conditions the third-law statement on a perfect crystal approaching absolute zero. Its wording is a standard introductory explanation, including the usual simplified disorder description. No material factual deterioration relative to the HF control is visible.
- **p4, French translation:** TT returns “Bonjour, comment allez-vous aujourd'hui ?” HF returns the same polite translation with a typographic apostrophe. Both finish normally; the final answer is in the requested language.
- **p5, Fibonacci:** The extended TT output includes the previously missing loop and final return. Both code blocks return `[]` for n<=0, `[0]` for n=1, append 1, then iterate from 2 to n-1 adding the previous two values and return the list. TT's eight-value example `[0, 1, 1, 2, 3, 5, 8, 13]` is correct; HF's ten-value example is also correct. This assessment is from reading the complete code, not executing generated code. The exact old 256-token TT prefix is preserved, so this extension directly resolves that earlier missing-loop/return observation.

## Haiku prefix caveat and cache capacity

A larger generation cap changes TT cache allocation in the current public `generate` path. For p0's 60 prompt tokens, the required cache is `60 + max_new_tokens - 1`, rounded up to a page of 32. The original 256-token request therefore allocates capacity 320, or 10 pages; the 1024-token request allocates capacity 1088, or 34 pages. Each p0 artifact records one page-table allocation, consistent with those fresh allocations. This source/counter inference does not require a hidden change to the prompt or sampler.

`optimized_decoder.py:737-748` selects a short-cache SDPA configuration when page-table width is below 16. The two capacities cross that branch: the old request uses the configured short `[8,2]` grid and k-chunk 32, while the larger request selects the general grid and starts from the configured k-chunk 128, reduced until it divides mapped capacity (64 for 1088). Thus these are not identical low-level numerical executions with only a later stop counter.

That difference is a **plausible explanation** for a near-tie greedy choice changing at p0 token 32. It is not a proven causal diagnosis: no paired same-capacity logit experiment isolated the SDPA branch, and other run-state/numerical differences have not been excluded. The evidence proves prefix nonidentity and successful output at the tested larger cap; it does not prove termination of the old branch or bitwise invariance across capacity policies.

The quality verdict can close for the recorded suite because the matched-cap extended p0 itself produces a correct completed answer, all other selected TT answers complete coherently, and no capacity-invariant token stream is claimed. The preserved 256-token evidence remains a documented budget limitation, especially for xhigh reasoning. A further same-capacity replay is needed only if that separate bitwise/causal guarantee becomes a requirement; it is not required to reinterpret a correct observed final haiku as a quality failure. No implementation change is supported solely by this free-running prefix mismatch.

## Shared degeneration checks

The shared `models/common/readiness_check/check_degenerate_output.py` module was imported directly by file path. Its `check_completion` function received saved text and token IDs for every original completion and every extension. With nonempty text, it evaluates word-based duplication/loop metrics. No Torch, TTNN, model or device module was imported.

- `qualitative_degeneracy.json`: all 12 original HF/TT outputs measured; exit 0, no critical or advisory findings.
- `qualitative_degeneracy_extended.json`: all eight extended HF/TT outputs measured; exit 0, no critical or advisory findings. Maximum adjacent-word duplication is 0.0161 for both TT and HF; maximum trigram-loop fraction is 0.0481 TT and 0.0759 HF. Extended p0 has zero adjacent duplication, with trigram-loop fractions 0.0435 TT and 0.0759 HF.

A clean machine check alone was insufficient at 256 tokens. The final verdict additionally depends on reading the delivered answers, their EOS boundaries, exact prompt controls and the explicit prefix caveat above. It is a six-prompt qualitative result, not a claim of universal reasoning reliability, identical HF/TT text, or completion of the HF story.

## Evidence identities

```text
tt_qualitative.json            bbb345afcba7797e5e1c735ed4c034a4e3e9b58e72d9422070aaed9c3f34ae98
hf_qualitative_256.json        ea77402b6404022c1bed0fe041aca374409dd7326d387a378f0d99d10f5a671b
tt_qualitative_extended.json   1bcabcb4bf7560220da059c504a3e72600ecf17ad184de608e011b5aeaf2d208
hf_qualitative_extended.json   61221a3a42f6d2ae431a13fd583c45c3dacc957e0aad20fc5cf6af938465fe39
check_degenerate_output.py     2629d86ae7be926a21e4839d151d21037e0851e98266f1b9a100e7cf14541db4
```
