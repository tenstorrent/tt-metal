# Shared qualitative-suite verdict

All six shared prompts were rendered with the pinned tokenizer's chat template
and generated greedily by exact-revision HF and optimized TT for 50 tokens.

Every TT output is coherent, prompt-relevant English and closely follows its HF
control's setup. There is no mechanical repetition, wrong-language drift,
prompt echo, leaked control token, cross-request leakage, corrupt first token,
or suspicious early semantic divergence. The 50-token budget mostly covers the
model's thinking preamble, so answer completeness is not claimed.

The haiku case brainstorms haiku concepts; supervised/unsupervised identifies
the requested comparison; story completion analyzes the supplied stem;
thermodynamics identifies the three-law task; French translation identifies
source and target; Fibonacci identifies Python and a suitable function. The
matched HF controls exhibit the same thinking-first style.
