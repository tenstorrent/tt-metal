# Qualitative suite verdict

Verdict: pass, with two classified observations.

Prompt mode `chat`, rendered with `tokenizer.apply_chat_template(add_generation_prompt=True)`
from `Qwen2Tokenizer` on the local snapshot; full metadata in
`qualitative_prompt_format.json`, rendered text and token ids per prompt in
`rendered_prompts.json`. Suite: the shared six-prompt set. Greedy, 256 new tokens per side.
Control: the same checkpoint at its native bfloat16 through `transformers.generate`.

## Answer quality

| prompt | TT | control |
| --- | --- | --- |
| p0 haiku | "Data flows in light / Patterns learn from hidden noise / Models find the way", a valid 5-7-5 | did not close `</think>` within the budget |
| p1 supervised vs unsupervised | correct, teacher analogy | same substance |
| p2 story completion | coherent, a brass key that fits no door | coherent, a brass key in a music box |
| p3 thermodynamics | correct first law, same phrasing as the control | same |
| p4 translate to French | `Bonjour, comment allez-vous aujourd'hui ?` | identical |
| p5 Fibonacci | did not close `</think>` within the budget | correct Python |

None of the failure modes the skill enumerates are present: no wrong language, no prompt echo,
no mechanical repetition, no doubled subwords, no cross-request leakage, no repeated or corrupt
first token, no gibberish. p4 had the shortest common prefix with the control at three tokens
and still produced the identical correct translation, so prefix length is not a defect signal
here.

## Observation 1: text continues past the stop token

TT output for p0 and p4 continues after `<|im_end|>` into `<|endoftext|><|im_start|>...`. This
is the harness, not the model: `generate` produces a fixed token count and does not stop on
EOS, so p4 ran the full 256 tokens where the control stopped itself at 113. Serving does not
inherit this, because vLLM owns stop-token handling. Reading it as control-token leakage would
be wrong.

## Observation 2: reasoning length differs from the control

p0 and p5 swap which side finishes thinking inside the budget: TT closes `</think>` on p0 where
the control does not, and the control closes it on p5 where TT does not. Both sides take valid
but different reasoning routes, which is the expected consequence of free-running greedy
decoding once the first token differs. It is not a correctness signal, and the teacher-forced
comparison recorded in `../doc/multichip_evidence.md` is the measurement that bears on accuracy.

## Scope

This covers prompt format and answer quality on six prompts at 256 tokens. It is not an
accuracy gate: no top-1/top-5/top-100 and no AIME24 reference were produced, and those remain
open for stages 6/7.
