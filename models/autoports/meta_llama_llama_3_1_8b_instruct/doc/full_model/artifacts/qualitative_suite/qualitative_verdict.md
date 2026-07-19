# Shared qualitative-suite verdict

Verdict: pass for all six prompts.

The exact Llama 3.1 Instruct `TokenizersBackend` rendered every prompt from
`models/common/readiness_check/vllm_prompts.txt` as a chat user turn with
`apply_chat_template(tokenize=False, add_generation_prompt=True)`, followed by
`encode(add_special_tokens=False)`. HF and TT both used greedy decoding for up
to 100 new tokens. The saved `qualitative_prompt_format.json` contains every
rendered prompt, token ID, first-divergence index, and trace-counter delta.

| ID | Prompt task | HF / TT tokens | First divergence | Manual TT verdict |
| ---: | --- | ---: | ---: | --- |
| 0 | Machine-learning haiku | 16 / 16 | 12 | Valid three-line haiku; clean EOS |
| 1 | Supervised vs. unsupervised | 100 / 100 | 4 | Clear labeled-data explanation, truncated only by cap |
| 2 | Story completion | 100 / 100 | 10 | Coherent fantasy continuation with stable entities/style |
| 3 | Laws of thermodynamics | 100 / 100 | 22 | Correct zeroth-law explanation and transition to first law |
| 4 | English-to-French translation | 77 / 100 | 25 | Correct translation and useful word-by-word explanation |
| 5 | Python Fibonacci function | 100 / 100 | 5 | Correct framing, recursive signature, docstring, and code start |

All TT completions are coherent, relevant, and in the requested language or
format. None shows wrong-language drift, stale-token duplication, phrase
looping, malformed special-token spill, or unexplained early termination.
Prompt 0 and HF prompt 4 terminate normally with `<|eot_id|>`; other endings
are ordinary 100-token truncations. The machine checker reports zero adjacent
duplication for all six TT outputs and trigram-loop fractions from 0.0375 to
0.2308, with no findings.

The first TT prompt was repeated after `reset()`: token IDs were bit-identical,
both decode trace IDs were reused, there was no recapture/release, the prefill
trace replayed once, and the unchanged page table required zero H2D copies.
Different later prompt shapes correctly replace incompatible prefill/decode
traces once per request; they do not perform per-token host feedback.

Artifacts:

- `qualitative_prompt_format.json`
- `degenerate_report.json`
- `prompt_00` through `prompt_05`, each containing rendered prompt, HF/TT
  completion, and `autoregressive_meta.json`
- `../../logs/full_model_qualitative_suite.log`
- `../../logs/check_degenerate_qualitative_suite.log`
