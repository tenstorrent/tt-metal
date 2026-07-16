# Stage 09 qualitative verdict

Pass for vLLM serving integrity, with base-checkpoint quality limitations.

The six raw-continuation prompts were read in both greedy and sampled form.
All outputs remain grammatical and in the expected language. There is no
gibberish, token-feedback corruption, wrong-language drift, cross-request state
leakage, or adjacent-token duplication. The scoped degeneracy checker exits
zero. There is nevertheless visible content-quality contamination: several
continuations become lists of unrelated requests, and some greedy outputs enter
phrase-level loops. Those are reported rather than treated as clean
instruction following.

The quality is not uniformly instruction-following:

- The haiku pair begins with an on-topic haiku, then continues into related
  poetry/story requests.
- Both supervised-learning outputs enumerate related questions rather than
  answering; greedy repeats the same question mechanically.
- The story pair begins coherently, then both continue into repeated or
  unrelated writing requests.
- The thermodynamics outputs stay in a science-question corpus trajectory but
  do not answer the requested laws; both are repetitive, especially the sampled
  biology/thermodynamics question list.
- The translation outputs continue a translation-exercise corpus and do not
  supply the requested translation.
- The Fibonacci outputs are on-topic: greedy provides a valid Python
  implementation and example, while sampled stops after describing the desired
  function and does not include code.

These failure modes match the Stage 08 Hugging Face/base-model controls rather
than indicating a vLLM serving regression.  In particular, the Hugging Face
greedy controls repeat the same supervised-learning and thermodynamics question
lists, and continue the same translation-exercise pattern.  The exact
`GemmaTokenizer` has no chat template, so Stage 09 intentionally records
`prompt_format=raw_continuation` instead of inventing an instruction format.
Request-list contamination and mechanical phrase repetition are therefore
documented base-checkpoint/prompt-format limitations. They are not presented
as strong instruction-following quality or hidden by the serving-integrity
pass.

Evidence:

- `vllm_qualitative_outputs.json`: all twelve served completions.
- `degenerate_output_check.json`: machine serving-integrity check.
- `../doc/datatype_sweep/qualitative/vllm_qualitative_outputs.json`: Stage 08
  Hugging Face and selected-TT controls (referenced from the model directory).
- `../doc/datatype_sweep/qualitative/verdict.md`: prior checkpoint-format
  assessment (referenced from the model directory).
