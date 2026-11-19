# Supplemental matched greedy inspection

The stage owner directly read all six TT greedy completions and all six pinned HF greedy controls from `tt_outputs.json` and `hf_outputs.json` after generation completed. This inspection is supplemental; the independent 120-example sampled blind comparison supplies the frozen comparative verdict.

| Prompt | Direct observation |
|---|---|
| q01, machine-learning haiku | Both spend the matched 128-token cap counting syllables in topical English reasoning. TT's “Patterns learn from data” is counted and reconsidered; HF considers “Data flows like streams.” Neither capped excerpt supplies a complete final haiku. |
| q02, supervised/unsupervised learning | Both explain labeled teacher examples versus unlabeled grouping. TT completes the response and emits the configured EOS; HF ends at the token cap during its second example. |
| q03, story continuation | Both continue the fairy tale with a small brass object and coherent narrative. Both excerpts are capped mid-story. |
| q04, thermodynamics | Both begin the first/second-law explanation and are capped during the second law. |
| q05, French translation | Both give “Bonjour, comment allez-vous aujourd’hui ?” after permitted English reasoning. |
| q06, Fibonacci Python | Both begin the same iterative-function approach, capped during the initial conditions. |

No observed greedy output contains mechanical doubled-token feedback, gibberish, unrelated cross-request material or a corrupt first token. `</think>` is the checkpoint's thinking boundary, and `<|im_end|>` is the recorded configured EOS retained inclusively by both control and TT trimming; these expected markers are not unexplained control-token leakage. Thinking and incomplete answers solely from the common cap are exclusions in the pre-existing frozen rubric. This is bounded evidence at 128 new tokens, not a long-answer completion guarantee.

HF control reuse is justified in `../../final_qualitative_plan.md` (stage-root plan) and the top-level `final_validation_v1.json` `qualitative_controls` copies/hashes: unchanged model revision, tokenizer/chat template, six prompts, settings, seeds and EOS handling. No new HF generation was run concurrently with device measurements.
