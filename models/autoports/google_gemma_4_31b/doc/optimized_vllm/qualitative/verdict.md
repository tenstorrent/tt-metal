# Qualitative verdict

Pass for serving integrity; weak instruction following is an inherited
base-checkpoint limitation.

All six greedy and six sampled raw continuations were reviewed. They are
grammatical, remain in the expected language, and show no gibberish,
cross-request contamination, stale-token feedback, or new pathological loop.
The mechanical degeneracy audit reports zero findings and exit code 0.

- Haiku: both trajectories begin with an on-topic haiku, then continue related
  prompt/completion examples.
- Supervised learning: both continue a related-question corpus; greedy repeats
  the same question, matching the raw standalone behavior.
- Story: both are coherent continuations; sampled is especially direct, while
  greedy later continues writing prompts.
- Thermodynamics: sampled begins directly explaining the laws; greedy continues
  a related-question corpus.
- Translation: both continue translation exercises rather than directly
  translating the requested sentence.
- Fibonacci: both explain the sequence and begin a useful implementation.

The prompt format is intentionally `raw_continuation`, because this base
tokenizer has no native chat template. The same continuation-style weaknesses
appear in the selected datatype-sweep standalone control. Therefore these
outputs validate the serving and async-feedback path but do not claim strong
instruction-following quality.

Artifacts:

- `qualitative_prompt_format.json`
- `qualitative_rendered_prompts.json`
- `vllm_qualitative_outputs.json`
- `standalone_control_outputs.json`
- `degenerate_output_check.json`
