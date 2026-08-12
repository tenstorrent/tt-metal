# vLLM qualitative verdict

Verdict: **pass for serving correctness, with base-checkpoint request
contamination documented**.

The exact `tiiuae/Falcon3-7B-Base` revision `bf3d7ed586cb22a921520e2d681a9d3d7642cde8`
has no tokenizer chat template. The shared six-prompt suite therefore used plain
`/v1/completions`; the compatibility chat template was not used for this verdict.
Both greedy and sampled outputs were read in full.

- All twelve completions begin coherently, address the requested topic, and
  remain in the expected language. The story, thermodynamics, translation, and
  Fibonacci prompts have directly relevant content.
- No completion shows a repeated-token loop, doubled subwords, gibberish collapse,
  wrong-language drift, prompt echo, corrupt first token, or evidence from another
  concurrently served request.
- The haiku greedy answer repeats the haiku once as an answer/explanation, but does
  not enter a mechanical loop.
- Several 256-token tails continue into unrelated training-document patterns:
  extra questions/answers after the haiku and Fibonacci tasks, and story-writing
  instructions after the sampled translation. This is request contamination in
  a base-model autocomplete, so those tails are not fully on-topic even though
  the requested answers themselves are correct.
- `hf_exact_qualitative_controls.json` uses the exact same raw prompts, 256-token
  lengths, greedy profile, and seeded 0.7/top-p 0.9 profile on Hugging Face BF16.
  Those controls likewise continue into extra ML sections, translation
  questions, and unrelated Python exercises. This establishes checkpoint
  behavior without dismissing the contamination.
- The earlier malformed fragments precisely at absolute position 256 were not
  accepted as base drift: they exposed a vLLM trace/RoPE capacity bug. After the
  request-horizon fix, all tails are syntactically formed and the degeneracy
  checker reports no finding.

The serving path is therefore qualitatively consistent with the exact HF
control. It should not be described as instruction-tuned chat quality. There is
no gibberish, wrong-language drift, mechanical repetition, or evidence of
another live request contaminating state. Raw output is in
`vllm_qualitative_outputs.json`; prompt metadata is in
`qualitative_prompt_format.json`.
