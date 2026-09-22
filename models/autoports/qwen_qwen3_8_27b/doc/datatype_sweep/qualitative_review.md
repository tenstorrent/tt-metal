# Selected precision qualitative review

The exact chat-template shared suite runs through the normal selected-policy
constructor with pinned HF revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
Rendered prompts, prompt token IDs, HF control text, TT text and native output
IDs are preserved in `tt_qualitative.json`, `tt_qualitative_extended.json`, and
`tt_qualitative_story_2048.json`. The packaged degeneracy check passes with no
findings; that does not by itself establish instruction correctness.

**Controlled quality limitation:** the haiku's last line scans six syllables
against the conventional five. Fixed S60/G1024, cache1088/history1023 controls
reproduce this exact 347-token stream with selected native replay, host-greedy
sampling, and BFP4/HiFi2 head precision. Baseline BFP8/HiFi2 repeats its earlier
418-token 5/7/5 answer. Thus the tested difference is sensitive to BFP4 head
weights; neither sampling delivery, replay nondeterminism nor LoFi fidelity alone
explains it. The supplied haiku prompt does not explicitly specify a syllable
count, but this report does not call the conventional meter satisfied.

The selected policy remains the fastest under the user's explicit full-model
top-1/top-5 gates. The six-prompt suite is not claimed universally instruction-
correct, and the meter limitation is retained rather than called repaired.
`AUTODEBUG_haiku.md` and `AUTOFIX_haiku.md` record the source diagnosis and
controlled experiments.

| Prompt | Final TT tokens | Direct inspection |
| --- | ---: | --- |
| 0: machine-learning haiku | 347 | “Data flows in light / Models learn from hidden signs / A new truth emerges” scans 5/7/6. HF and Stage7 controls scan 5/7/5. Controlled BFP4-weight-sensitive difference. |
| 1: supervised/unsupervised learning | 144 | Correct labeled-target versus unlabeled-pattern distinction, with suitable photo and customer examples. |
| 2: story completion | 1452 | Mira's key leads to the Valley of Lost Sounds; the repaired bell returns sounds and the story ends coherently. Exact 1024-token prefix agrees with the shorter TT control. |
| 3: thermodynamics | 268 | Correct energy conservation with ΔU=Q−W convention, nondecreasing isolated-system entropy, and perfect-crystal zero-temperature limiting entropy. |
| 4: French translation | 115 | “Bonjour, comment allez-vous aujourd'hui ?” is a correct translation. |
| 5: Fibonacci | 353 | Correct function returning the first n values; separate example prints the first ten. Restricted host execution passes n=-1/0/1/2/8. |

All six final outputs reach EOS. The initial 256-token suite truncates several
reasoning/answer sequences; HF controls also truncate. A 1024-token extension
completes haiku, thermodynamics and code. Both HF and TT story controls still
truncate at 1024. The TT-only 2048-budget completion reaches EOS at1452 and
preserves the exact first1024 tokens, proving the earlier cutoff is budget,
not a token-feedback stall. This longer TT result is completion evidence, not
an equal-budget comparison with the 1024-token HF control.

Raw reasoning before `</think>` and EOS markers remain in both TT and HF
artifacts. They are retained diagnostic output, not postprocessed user text.
No wrong language, mechanical looping, doubled subwords, malformed first token,
or cross-request leakage was found in the inspected outputs. Creative narrative
and reasoning length differ from HF; that difference alone is not token-for-token
quality accuracy. The haiku syllable error is separately tracked above.

The host code checker originally assumed the generated list was named `fib`.
This correct answer uses `sequence`; the checker now permits `.append` only on
a locally initialized list while retaining restricted imports/calls and actual
sequence-result tests. `qualitative_check.log` and `qualitative_metrics.json`
record the passing checker, source hashes, completion counts and story prefix.
The independent reviewer additionally executed both HF and TT Fibonacci controls
for n=-1/0/1/2/10 and inspected the completed story directly.
