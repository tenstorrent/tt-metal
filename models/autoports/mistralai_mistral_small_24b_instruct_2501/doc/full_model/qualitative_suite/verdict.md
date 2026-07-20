# Shared qualitative-suite verdict

Verdict: pass. All six prompts use the exact checkpoint's corrected Mistral
tokenizer and chat template. HF and TT receive identical rendered prompt token
IDs and greedy controls. Four prompts run the full 128-token budget on both
sides; translation runs 62/62 tokens and the haiku ends normally at 16 HF / 18
TT tokens. A repeated TT run of prompt 1 is token-exact deterministic.

- `shared_01`: both are fluent three-line machine-learning haiku. HF ends
  “Knowledge from chaos”; TT ends “Knowledge blooms anew.” Both terminate with
  EOS and have no formatting/control leakage.
- `shared_02`: both correctly distinguish labeled supervised learning from
  pattern discovery without labels, using simple teaching analogies. Both are
  cut at the 128-token budget while beginning the unsupervised example; neither
  loops or drifts.
- `shared_03`: both continue the requested story coherently with inventor Eli,
  a workshop, and an energy-bearing crystal. They diverge in harmless narrative
  wording and stop only at the fixed budget.
- `shared_04`: both correctly state the zeroth and first thermodynamic laws and
  begin the same `ΔU = Q - W` explanation before the fixed budget. There is no
  factual or language drift in the visible completion.
- `shared_05`: both give the same correct French translation, “Bonjour, comment
  ça va aujourd'hui?”, then a coherent English breakdown and EOS. French text
  is requested behavior, not wrong-language drift.
- `shared_06`: both explain Fibonacci and begin a valid Python function with
  the same edge-case structure. Both are cut at 128 tokens, without doubled
  subwords, prompt echo, or code corruption.

The scoped `check_degenerate_output.py` run reports no finding across this
suite plus the original autoregressive artifact. Adjacent duplication is zero
for five suite outputs and 0.0094 for the story, far below the 0.10 critical
threshold. There is no single-token collapse, phrase loop, repeated/corrupt
first token, gibberish, cross-request leakage, or mechanical repetition.

Artifacts: per-prompt rendered prompt, metadata, token IDs and HF/TT texts;
`qualitative_prompt_format.json`; `suite_summary.json`;
`degenerate_output_report.json`; and `../logs/check_degenerate_output.log`.
