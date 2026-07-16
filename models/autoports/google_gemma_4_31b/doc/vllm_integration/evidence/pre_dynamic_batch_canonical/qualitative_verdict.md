# Stage 09 qualitative verdict

Pass for vLLM serving integrity, with base-checkpoint quality limitations.

The six raw-continuation prompts were read in both greedy and sampled form.
All outputs remain grammatical, in the expected language, and recognizably
related to the prompt or to the checkpoint's prompt-corpus continuation style.
There is no gibberish, token-feedback corruption, wrong-language drift, or
cross-request contamination.  The scoped degeneracy checker exits zero and
finds no adjacent-duplication failure; its detailed measurements are in
`degenerate_output_check.json`.

The quality is not uniformly instruction-following:

- The haiku pair begins with an on-topic haiku, then continues into related
  poetry/story requests.
- The supervised-learning sampled output gives a clear and correct explanation;
  greedy output mechanically repeats the related question.
- The greedy story is coherent and complete.  The sampled story starts on topic
  and then continues into a list of related story-writing requests.
- The thermodynamics outputs stay in a science-question corpus trajectory but
  do not answer the requested laws; both are repetitive, especially the sampled
  biology/thermodynamics question list.
- The translation outputs continue a translation-exercise corpus and do not
  supply the requested translation.
- The Fibonacci outputs are correct: greedy explains the sequence, and sampled
  provides a valid Python implementation and example.

These failure modes match the Stage 08 Hugging Face/base-model controls rather
than indicating a vLLM serving regression.  In particular, the Hugging Face
greedy controls repeat the same supervised-learning and thermodynamics question
lists, and continue the same translation-exercise pattern.  The exact
`GemmaTokenizer` has no chat template, so Stage 09 intentionally records
`prompt_format=raw_continuation` instead of inventing an instruction format.
Request-list continuation and mechanical phrase repetition are therefore
documented base-checkpoint/prompt-format limitations.  They are not presented
as strong instruction-following quality.

Evidence:

- `vllm_qualitative_outputs.json`: all twelve served completions.
- `degenerate_output_check.json`: machine serving-integrity check.
- `../doc/datatype_sweep/qualitative/vllm_qualitative_outputs.json`: Stage 08
  Hugging Face and selected-TT controls (referenced from the model directory).
- `../doc/datatype_sweep/qualitative/verdict.md`: prior checkpoint-format
  assessment (referenced from the model directory).
