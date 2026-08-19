# Autoregressive Qualitative Verdict

- Model: `Qwen/Qwen3.6-35B-A3B`
- Prompt: `models/common/readiness_check/autoregressive_prompt.txt`
- Prompt format: raw completion prompt with tokenizer special tokens
- Generated tokens: HF `100`, TT `100`
- TT trace: traced greedy token-out path
- Degenerate-output check: pass

Manual verdict: pass.

The TT completion is coherent English and stays on the requested story prompt.
It diverges from the HF greedy reference after the shared opening, which is
expected for free-running autoregressive generation. The divergence is not a
decode-loop bug signature: the machine checker reports adjacent duplication
`0.0`, trigram loop fraction `0.038`, no critical findings, and informational
HF/TT token agreement `14/100`.

Artifacts:

- `autoregressive_meta.json`
- `hf_completion.txt`
- `tt_completion.txt`
- `degenerate_output_report.json`
