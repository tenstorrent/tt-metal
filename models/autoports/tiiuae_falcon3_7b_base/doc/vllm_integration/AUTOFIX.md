# AutoFix Report

## Starting evidence

- Diagnosis: `AUTODEBUG.md` in this directory. The fresh runner launched but
  could not execute repository reads because its nested sandbox lacked `bwrap`;
  the delegated fresh-context agent completed the required read-only fallback.
- Original failure: `readiness_vllm/server.log` rejected vLLM's greedy
  `temperature=0, top_k=vocab_size` representation and then exposed missing
  host-compatibility and output-format boundaries.

## Hypothesis experiments

- Hypothesis: vLLM's greedy representation must map to canonical top-1 before
  Falcon's split sampler.
  Experiment: rerun the shared smoke profile after exact zero-temperature
  normalization.
  Result: the previous `top_k` exception disappeared and execution advanced.
  Verdict: verified.
- Hypothesis: `sampling_params=None` signals explicit host compatibility and
  requires logits, not device tokens.
  Experiment: run `test_min_p` and mixed-parameter serving directly against a
  held server after delegating to the generator's host prefill/decode modes.
  Result: both tests passed.
  Verdict: verified.
- Hypothesis: synchronous prefill device sampling must return a host token
  vector to vLLM while preserving the device token for decode feedback.
  Experiment: rerun mixed-parameter serving after reading only sampled token
  IDs at the prefill completion boundary.
  Result: prefill and traced decode completed; the test passed.
  Verdict: verified.
- Hypothesis: host decode logits require `[B,1,V]` for this plugin version.
  Experiment: rerun `test_min_p` after adding the singleton sequence axis.
  Result: all ten min-p requests passed.
  Verdict: verified.
- Hypothesis: the base checkpoint needs an explicit compatibility chat template
  for the shared chat-only logprobs node.
  Experiment: rerun the full smoke profile with `base_chat_template.jinja`.
  Result: 3 passed, 1 skipped by the plugin's supported max-logprobs cap.
  Verdict: verified.

## Final status

Fixed for the smoke gate. The successful command used the shared
`run_vllm_server` runner with `--sampling-profile smoke`, `--max-num-seqs 1`,
`--max-model-len 32768`, `--block-size 32`, on-device sampling mode `all`, and
the explicit base-model chat compatibility template. Full sampling, larger
capacity, qualitative, and benchmark gates remain stage work.
