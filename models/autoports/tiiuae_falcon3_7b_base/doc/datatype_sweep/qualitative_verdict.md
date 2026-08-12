# Selected-policy qualitative verdict

Falcon3-7B-Base has no tokenizer chat template, so the shared six-prompt suite
correctly uses base-model completion prompts. The selected policy ran 100-token
HF and TT controls through the normal traced generator path. The degeneracy
checker passed all prompts; TT exactly matched HF for the Fibonacci prompt and
produced coherent, on-topic continuations for the other five.

The haiku prompt repeats a complete haiku stanza. This is controlled rather
than a new dtype regression: the completed full-model stage produced the same
stanza repetition under the same selected BFP4/LoFi policy, while the current
run has zero adjacent-token duplication and no corrupt/control tokens. No wrong
language, prompt echo, doubled subwords, token leakage, cross-request leakage,
or gibberish was observed.

Artifacts: `results/selected_bf16_kv_qualitative/qualitative_prompt_format.json`,
the six prompt directories, `prompt_verdict_inputs.json`,
`degenerate_output.json`, and `results/selected_bf16_kv_qualitative.log`.
