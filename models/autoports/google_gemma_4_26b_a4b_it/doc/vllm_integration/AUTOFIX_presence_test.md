# Presence-penalty test control

The initial full profile failed both presence tests because every penalty on the
raw cyclic prompt `a b c a b c a b c` produced the same40-token continuation.
The assertions require different text, which is not guaranteed when all likely
next tokens receive the same one-time penalty.

`readiness_vllm/presence_control.json` records full-model device and explicit
vLLM CPU(logprobs3) paths at presence0/2. All four40-token sequences are exactly
equal. This independently reproduces the original output behavior with the
canonical vLLM sampler. These are raw completion sampling mechanics tests, not
prompt-correct qualitative evidence.

`presence_prompt_control.json` preserves two less constrained candidate prompts.
For `The story begins: `, presence0 and2 produce different token sequences;
each exactly matches the corresponding CPU(logprobs3)40-token control. The
shared tests now use this prompt, retaining all deterministic-group and variation
assertions, penalty values, batch sizes and output lengths. No model, sampler or
penalty math was changed. The live targeted class result is recorded separately
in `presence_live_retry.log`; final full-profile rerun remains required.
