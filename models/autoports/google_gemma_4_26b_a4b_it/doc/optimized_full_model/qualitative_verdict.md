# Qualitative verdict

All six shared chat-template prompts were rerun through all30 layers, in both
EOS-stopping mode and buffered128-token mode. The buffered prefix exactly
matches the EOS-stopping output. All six EOS-stopping token sequences exactly
match the accepted Stage06 outputs (`qualitative_comparison.json`). The same
revision's HF outputs and rendered prompt/token metadata are embedded in
`qualitative_tt.json`; controls originate from `../full_model/qualitative_hf.json`.

Actual HF and TT text was read. shared_0 gives coherent haiku ("Data flows like
rain"); shared_1 explains supervised learning with labeled fruit; shared_2
continues an inventor story; shared_3 correctly introduces conservation of
energy; shared_4 gives valid formal/informal French; shared_5 begins an iterative
Fibonacci implementation. Long answers/code truncate at128 tokens in both HF
and TT controls; this cap is a test limit, not a model capability reduction.
The exposed `<turn|>` in raw EOS-preserving artifacts is the expected stop token,
not unintended control text in skip-special-token output. No doubled tokens,
wrong-language drift, collapse, or cross-request leakage was observed.

The4096-token repeated-document workload is separately labeled raw continuation
stress. Repetition there mirrors the repeated input and is not the chat-quality
verdict. It must not replace this prompt-correct suite.
