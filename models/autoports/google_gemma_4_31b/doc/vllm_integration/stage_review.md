# Stage 09 final independent review

Verdict: `clean-pass`

Required work: none.

The fresh read-only review compared the live main and nested-vLLM diffs with
the complete Stage 09 goal and the `vllm-integration`, `tt-device-usage`,
`tt-enable-tracing`, `qualitative-check`, and `stage-review` contracts. It
inspected the adapter, generator, multichip decoder, readiness runner, plugin
registration and sampling changes, adapter/plugin tests, context contract and
capacity derivation, final runner/server/sampling logs, prompt checks,
qualitative outputs and verdict, degeneracy result, benchmarks, runtime audit,
cleanup evidence, and documentation.

## Evidence accepted

- Shared TT vLLM registration selects
  `models.autoports.google_gemma_4_31b.tt.generator_vllm:Gemma4ForCausalLM`.
- The adapter uses the selected precision policy and vLLM-owned hybrid cache,
  with async nonblocking canonical split model/sampler traces and device-greedy
  token feedback. Optional host compatibility is explicit and does not replace
  the measured path.
- The direct non-aligned request passes at 149 input tokens.
- Two real `113279`-input plus one-output requests pass at advertised
  `max_model_len=113280`; the adjacent aligned candidate `113344` is source
  proven short by `148800` bytes/bank.
- Full shared sampling reports `72 passed, 1 skipped`; the skip is the explicit
  all-vocabulary logprobs capability probe, while bounded logprobs and all
  serving-correctness cases pass.
- All twelve final raw-continuation outputs were read and honestly classified.
  Base-checkpoint request-list continuation and repetition match Stage 08
  controls; there is no serving-only gibberish, wrong-language drift, request
  leakage, or token-feedback corruption. The scoped degeneracy check exits 0.
- Primary 127/128/1: TTFT `992.586 ms`, TPOT `38.023 ms`, ITL P50/P99
  `29.348/29.739 ms`, throughput `21.974 tok/s`, decode `26.300 t/s/u`.
- Secondary CI 99/100/32: TTFT P50/P99 `8485.248/8488.457 ms`, TPOT
  mean/P99 `77.373/127.442 ms`, ITL P50/P99 `55.807/687.715 ms`, throughput
  `201.070 tok/s`; it is correctly not used as the headline decode rate.
- The final server terminated cleanly, all devices closed healthy, and no live
  process holds hardware. Historical PID-1 zombies do not hold descriptors.

## Nonblocking observations

- The final server emits the classified cooperating split-trace allocator
  warning and nanobind leak diagnostics during interpreter shutdown. All gates
  complete, UMD closes every device, and post-run hardware/process audits pass.
- Firmware bundle 19.9 is newer than the fully tested 19.5 bundle; no runtime
  failure or recovery occurred.
- `113344` was not knowingly executed because the exact source/allocator model
  proves a hard physical shortfall. The passing boundary request and adjacent
  source proof satisfy the physical-limit requirement.

The earlier more-work verdict is preserved as `stage_review_pre_max_context.md`;
`stage_review_initial.md` preserves the first review.
