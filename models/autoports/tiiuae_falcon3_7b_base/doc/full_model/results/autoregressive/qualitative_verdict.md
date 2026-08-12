# Autoregressive qualitative verdict

- Prompt: common readiness story-completion prompt, 59 tokens.
- Budget: 100 HF tokens and 100 TT tokens.
- First divergence: generated-token index 5; indices 0 through 4 match.
- HF: coherent English story continuation, relevant to Elena and the unusual
  sunlight; no repetitive loop or wrong-language drift.
- TT: coherent and grammatical English story continuation about dancing
  sunlight, rainbows, and Elena telling her grandmother; no repetitive loop,
  wrong-language drift, malformed special token, or premature EOS.
- Verdict: pass. Divergence is expected for a free-running approximate model;
  both branches remain locally and globally coherent for the full budget.

Artifacts read during review: `hf_completion.txt`, `tt_completion.txt`, and
`autoregressive_meta.json` in this directory.
