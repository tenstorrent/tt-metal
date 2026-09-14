# Final qualitative review

The final full64-layer shared suite passes the packaged degeneracy checker and
all six prompts have coherent EOS-complete outputs. Source hashes, exact
completion lengths, EOS flags and the host-check command are in
[qualitative_metrics.json](qualitative_metrics.json). The pinned model revision,
chat-template metadata, rendered/tokenized prompts, raw TT output and HF controls
are in [tt_qualitative.json](tt_qualitative.json) and
[tt_qualitative_extended.json](tt_qualitative_extended.json).
Raw reasoning and special-token markers are intentionally retained for both
implementations; they are not user-visible text postprocessing or prompt echo.

| Prompt | Final TT tokens | Inspection |
| --- | ---: | --- |
| 0: haiku | 418 | “Data flows in light / Models learn from hidden signs / Clarity forms now” is coherent and follows5/7/5 syllables. |
| 1: supervised versus unsupervised learning | 148 | Correct distinction between labeled targets and structure found without labels. |
| 2: creative story | 1700 | Elara's brass key, mirror and encounter with a kind king form a coherent completed story. See the controlled budget investigation below. |
| 3: thermodynamics | 380 | Correct first, second and third laws with appropriate concise explanation. |
| 4: French translation | 145 | Correct French translation, no wrong-language continuation. |
| 5: Fibonacci function | 337 | Complete Python function. Restricted host execution verifies n=-1/0/1/2/8, including empty and short sequences. |

The256-token suite initially cut off several reasoning/answer sequences. A
1024-token extension completed prompts0/3/5. Both TT and HF story controls still
hit1024 tokens; TT spends more tokens planning. The separate
[2048-token story run](tt_qualitative_story_2048.json) completes at1700 tokens.
Its first1024 token IDs exactly equal the shorter TT run, confirming a budget
cutoff rather than a token-feedback stall. The copied HF control still has a
1024-token budget: the longer TT result establishes completion, not an equal-
budget HF comparison. The stage6 story ended earlier; creative output length
is not token-for-token accuracy, and the current completed story is coherent.

Fresh AIME24 chat-template [autoregressive artifacts](autoregressive) accompany
[readiness_final.json](readiness_final.json). Autoregressive divergence from HF
is distinct from the standardized teacher-forced ranking gate: prefill and
decode both have top5=top100=100% across100 positions. No mechanical looping,
wrong-language output or cross-request leakage was found in the inspected
shared suite. Fixed S128 benchmark prompts are performance fixtures and are
not used as qualitative evidence.
