# Qualitative verdict

Verdict: pass. The TT full model produces coherent, prompt-relevant 100-token completions without degeneracy, wrong-language drift, prompt/control-token leakage, or trace-specific divergence.

## Canonical autoregressive story

HF and TT share the first five generated tokens and diverge at token index 5. HF continues with a red bird named Solara; TT continues with a glowing mountain and an explanatory encounter. Both are grammatical English stories consistent with the village prompt. Neither loops, repeats a short phrase, changes language, exposes control tokens, or collapses early. Early free-running divergence is compatible with the 91% teacher-forced top-1 and 99% top-5 measurement; qualitative equivalence does not require identical continuations.

The canonical degeneracy checker independently reports no finding: adjacent duplication is 0.0 and trigram-loop fraction is 0.039 across the 100-token TT story (`results/autoregressive/degenerate_output.json`).

## Six-prompt shared suite

| Prompt | TT review |
|---|---|
| Machine-learning haiku | Produces a coherent three-line poem and explanation; later resumes a dataset-like question, a base-model autocomplete quirk rather than corruption |
| Supervised vs unsupervised | Correctly distinguishes labeled and unlabeled learning, then continues into related clustering material |
| Inventor story | Coherent narrative; the invented machine name `1984` is repeated as a proper name but does not become a token loop |
| Thermodynamics | Correctly states the Zeroth and First Laws with relevant explanation |
| English to French | Gives the requested French translation and stays in relevant bilingual explanation |
| Fibonacci Python | Produces a plausible iterative Python function with correct base cases |

`check_degenerate_output.py` reported no findings for all six TT completions. Adjacent duplication was 0.0 for every prompt; measured trigram loop fractions remained low and reflected normal prose/code reuse.

## Trace control

For a 100-token haiku control, host eager greedy, traced split greedy, traced greedy after reset/recapture, and traced teacher-forcing were identical for all 100 tokens, with no first-divergence index. Final cache and rotary positions were both 107. A seeded stochastic trace produced a distinct but coherent machine-learning haiku and explanation. This isolates the visible base-model continuation quirks from trace, sampler, cache, and token-feedback correctness.
