# Autoregressive qualitative verdict

Verdict: pass.

The input is AIME24 prompt index 0 rendered as one Llama 3.1 user turn with
`add_generation_prompt=true`.  It contains 184 tokens.  A rerun of the HF
side with an explicit all-ones attention mask reproduced all 100 saved HF
tokens exactly, so the original pad-equals-EOS warning did not alter the
artifact.  The readiness reference and teacher-forcing comparison are
shifted left correctly: the logit at `prompt_len - 1 + i` predicts generated
token `i`.

HF and TT share the first six generated tokens, `To solve this problem, we`,
and first diverge at generated-token index 6.  The divergence is an ordinary
greedy branch, not a stale-token symptom: traced top-5 is 100%, device token
feedback matches the last sampled output, and current-position and RoPE state
advance by exactly one on every replay.

The TT completion remains a coherent English mathematical response.  It
identifies the need to set up equations from the two travel-time conditions,
uses the supplied variables consistently, and continues step-by-step until
the 100-token cap.  It has no wrong-language drift, adjacent duplication,
phrase loop, malformed special-token spill, or premature EOS.  The machine
degeneration checker reports adjacent duplication 0.0, trigram-loop fraction
0.075, and 14 position-wise matching tokens out of 100, with no finding.  Both
HF and TT outputs end mid-derivation
because the comparison intentionally caps each side at 100 new tokens; that
is not an early model termination.

Artifacts:

- `hf_completion.txt`
- `tt_completion.txt`
- `autoregressive_meta.json`
- `degenerate_report.json`
