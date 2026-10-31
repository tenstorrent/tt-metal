# Selected-policy qualitative review

Selected-default outputs are in selected_default/qualitative_tt.json, with the same six pinned HF controls and prompt-format metadata copied alongside them. Prompts use the checkpoint's chat template; tokenizer and model revision remain842da3794eaa0b77d5f08bae87a17459d91ff475. This is a prompt suite, not a scored dataset evaluation.

- shared_0: coherent5/7/5 haiku (“Data feeds the code … The machine now learns”). The third line differs from HF, without visible degeneration.
- shared_1: correct supervised-versus-unsupervised explanation using a teacher/discovery analogy. Output reaches128 tokens mid-example; HF is also length-limited.
- shared_2: coherent continuation about a clockwork heart and inventor Elara; no prompt echo or mechanical loop. Both TT and HF continuations are capped.
- shared_3: correct introduction to thermodynamics, including the zeroth law and energy conservation. The128-token excerpt is incomplete, as is the HF control; no complete three-law answer is claimed.
- shared_4: “Bonjour, comment allez-vous aujourd'hui ?” is a correct concise French translation. It differs from the longer HF explanation but is not worse behavior. It stops at the pinned EOS.
- shared_5: coherent Fibonacci implementation explanation and code prefix. Both controls are capped at128 tokens; the partial function is not claimed to be executable complete code.

Visible `<turn|>` in the saved decoded text is the intentionally retained stop token, also present in HF artifact text. This is not evidence of serving token leakage; no serving endpoint is part of this stage.

The32,083-token agent-shaped conversation asks for a label retained from the first turn. All completed precision candidates retrieve ORCHID-742. They wrap the correct JSON in a Markdown fence despite “only JSON.” Higher-precision BFP8 decoder controls reproduce the exact completion, controlling this behavior against the dtype change. The refreshed baseline in baseline_bfp4_lofi/long_prefix.json also reproduces the exact completion. This check establishes retention and absence of visible free-running degeneration, not strict tool-call conformance or a long-context HF accuracy score.
