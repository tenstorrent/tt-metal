# Qualitative inspection

All six shared readiness prompts were rendered with the pinned HF tokenizer's
chat template and run greedily for at most 128 new tokens on HF BF16 and the
complete 30-layer TT model. Sources: `qualitative_hf.json`, `qualitative_tt.json`,
`qualitative_divergence.json`; commands in `work_log.md`.

The actual HF and TT completions were read. Both haikus are coherent poems;
both supervised-learning answers explain learning from labeled examples;
both lost-compass stories remain coherent; both thermodynamics answers explain
the first law; both French translations are appropriate; both Fibonacci answers
produce Python explanations/code, truncated mid-code at the imposed token cap.
No mechanical repetition, token doubling, collapse, or wrong-language drift
was observed. The turn-end token follows the HF format.

Exact wording differs. At the first differing token, the TT choice ranks 2, 3,
7, 2, 2 and 2 under HF logits evaluated on the identical prefix, respectively.
The story's rank-7 choice begins a hyphenated compass description; the resulting
story remains coherent. This is an acknowledged lexical divergence, not an
exact-output claim. These six prompts are qualitative coverage, not a benchmark
score. The 100-position AIME teacher-forcing accuracy is reported separately.

The standard `run_autoregressive` comparison also passes. Actual completions in
`autoregressive/hf_completion.txt` and `autoregressive/tt_completion.txt` were
read: both introduce white light as a mixture of rainbow colors in accessible
English. HF starts with playing with a flashlight/prism; TT starts with holding
a flashlight and describes the atmosphere. The first differing token is index3
(zero-based), a lexical choice leading to different coherent explanations.
Both stop at the128-token budget mid-explanation, so this does not establish a
complete answer about Rayleigh scattering. Neither shows repetition, collapse,
wrong-language drift or token-feedback corruption. `degeneracy.json` reports
no findings, exit0; informational exact token agreement is8/128, not an
accuracy gate. The pinned HF BF16 reference uses the identical28-token chat
prompt recorded in `autoregressive/autoregressive_meta.json`.
