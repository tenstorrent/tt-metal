# Stage 09 qualitative verdict

Pass for vLLM serving integrity, with substantial base-checkpoint
instruction-following limitations.

All six raw-continuation prompts were read in both greedy and sampled form.
The twelve outputs are grammatical, coherent at the local sentence level, and
stay in the expected language. There is no gibberish, wrong-language drift,
cross-request state corruption, stale-token feedback, or unexplained token
duplication. `degenerate_output_check.json` has no findings and the checker
exited 0.

This is not a strong instruction-following result. The base model frequently
continues a corpus of related requests instead of answering the given request:

- Haiku: both paths first produce a coherent, on-topic haiku. Greedy then
  continues a list of related haiku requests and answers; sampled continues
  with an AI poem and robot-story prompt/answer sequence.
- Supervised versus unsupervised learning: greedy repeats the same question
  mechanically; sampled emits a list of other machine-learning questions.
  Neither provides the requested explanation.
- Story: both begin with a plausible completion. Greedy then loops between two
  writing requests; sampled continues a longer list of unrelated creative
  writing requests instead of sustaining the story.
- Thermodynamics: both remain broadly on topic but enumerate related questions
  rather than answering the three laws. Sampled eventually repeats the Gibbs
  free-energy/entropy question.
- Translation: both continue a translation-exercise request list and never
  supply the requested French translation. They do not drift into an incorrect
  natural language; the failure is request continuation rather than answering.
- Fibonacci: both are on topic and useful. Greedy returns a valid nth-number
  implementation; sampled returns a valid first-n-values implementation.

The repeated question lists, translation-exercise continuation, and weak raw
instruction following match the Stage 08 Hugging Face/base-model controls.
The exact tokenizer has `chat_template=None`, so Stage 09 intentionally uses
and records `prompt_format=raw_continuation` rather than inventing a chat
template. These trajectories therefore support serving-path integrity while
documenting checkpoint/prompt-format quality limits; they are not presented as
good instruction following.

Evidence:

- `vllm_qualitative_outputs.json`: all twelve final served continuations.
- `degenerate_output_check.json`: scoped mechanical serving-integrity check,
  zero findings and exit 0.
- `../doc/datatype_sweep/qualitative/vllm_qualitative_outputs.json`: Stage 08
  Hugging Face and selected-TT raw-continuation controls.
- `../doc/datatype_sweep/qualitative/verdict.md`: prior checkpoint-format
  assessment.
