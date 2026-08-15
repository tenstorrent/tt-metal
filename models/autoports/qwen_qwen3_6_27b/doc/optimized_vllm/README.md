# Qwen3.6-27B optimized vLLM serving

**Primary single-user TT plugin serving (`max_num_seqs=1`, 128 input / 128 output, one request, concurrency 1, temperature 0, `max_model_len=262144`, TP4 P300x2): TTFT P50/P99 3,784/3,784 ms; mean TPOT 61.893 ms; ITL P50/P99 55.840/56.850 ms; aggregate output 10.992 tok/s; TPOT-derived decode 16.157 t/s/u.**

This is the real vLLM TT plugin path through `tt/generator_vllm.py`. Relative
to the same integration baseline it improves TTFT by 8.6%, TPOT by 12.5%,
aggregate output throughput by 12.7%, and decode by 14.3% (14.138 to 16.157
t/s/u). It reaches 92.5% of the comparable optimized-full-model canonical
split-token result, 17.467 t/s/u.

Secondary CI capacity evidence (`max_num_seqs=32`, 100 input / 100 output, 32
requests, unconstrained admission, temperature 0, the same context, mesh,
trace, sampler, and TT precision config): TTFT P50/P99 162,573/162,574 ms;
mean TPOT 279.381 ms; ITL P50/P99 244.131/560.379 ms; aggregate output 17.049
tok/s. Its 3.579 TPOT-derived t/s/u is recorded only as a burst statistic, not
as headline decode performance.

## What changed

Steady decode previously reformatted and uploaded top-k, top-p, temperature,
and four penalty tensors on every scheduler step even when sampler state had
not changed. The adapter now keys the scheduler-owned sampler contract and
refreshes those persistent device tensors only when that contract changes.
Seed advancement remains owned by `SeedManager` and is deliberately independent.

AutoFix also found a pre-existing async alias in the plugin's unsupported-host-
feature compatibility path: submitted sampling work held the canonical
`torch.Generator` while later scheduling advanced it. The plugin now clones the
submitted generator and advances only canonical state. The focused regression
test passes, and the complete live plugin suite passes 72 tests with one skip.

## Serving and trace contract

- vLLM reports asynchronous scheduling enabled and the TT runner reports
  `trace_mode=all`, `sample_on_device_mode=all`.
- `decode_forward(..., read_from_device=False)` returns a TT device tensor.
  `token_out_decode_step` calls `ttnn.execute_trace(..., blocking=False)`;
  `read_decode_output(async_read=True)` initiates `.cpu(blocking=False)`, and
  the plugin synchronizes only at its deferred result boundary.
- The adapter delegates sampling to the full-model greedy split sampler. The
  measured path has no host argmax/top-1, full-logits readback, force-argmax,
  or generic eager sampler.
- Token, current-position, RoPE, page-table, cache, and sampler inputs remain
  persistent device tensors. Scheduler-owned sampler tensors now refresh only
  on state changes; seed state advances once per token.
- The inherited direct adapter stale-input gate covers changed token/current
  position, unchanged page table, and slot remap with zero host refresh/readback
  counters. The optimization does not alter token, position, page-table, or
  cache refresh logic. Full live sampling additionally covers mixed parameters,
  penalties, seeded batch ordering, and request isolation.
- `max_model_len=262144` exactly preserves `doc/context_contract.json`; exact
  65-token serving and reduced 65/63 mixed-slot evidence preserve non-aligned
  prompt support.

No Tracy, tt-perf-report, live-server device profiler, adapter profiler, or
`ReadDeviceProfiler` was collected for this stage. Evidence is benchmark JSON,
sampling output, qualitative output, trace/source contracts, and stale-input
checks.

## Evidence

- `artifacts/before/`: exact integration baseline primary and CI JSON/logs.
- `artifacts/after_maxseq1/`: final exact single-user JSON/log/server evidence.
- `artifacts/after_maxseq32/`: full sampling, raw-completion diagnostic,
  max-32 control, and CI burst JSON/log/server evidence. The raw qualitative
  file is not used for the quality verdict.
- `artifacts/after_chat/`: final optimized `/v1/chat/completions` greedy and
  fixed-seed sampled outputs, checkpoint-template rendered prompts and exact
  token IDs, plus the passing degeneration check.
- `artifacts/autofix_seed_rng/`: the isolated failing live suite and server log.
- `../vllm_integration/logs/reduced_target_stale_input.log`: direct persistent-
  input and non-aligned adapter trace evidence inherited unchanged.
- `AUTOFIX.md`: hypothesis, isolation, fix, and rerun result.

All successful servers terminated cleanly; final process audits found no
`vllm.entrypoints`, `EngineCore`, or readiness-runner process.

## Qualitative verdict

The final optimized chat-correct suite uses the same six shared prompts through
the checkpoint template, with greedy and sampled seed 20260815 outputs capped at
256 tokens. All 12 outputs were read. They are coherent, on-topic, grammatical,
and free of wrong-language drift, request contamination, adjacent-token
duplication, or repetition loops. The machine degeneration checker exits 0.
They match the healthy reasoning-first behavior in the datatype/full-model and
prior vLLM controls. The real limitation remains presentation: every response
reaches the 256-token cap during its visible reasoning and therefore does not
reach the concise final answer. The earlier fresh raw-completion diagnostic had
several malformed sampled phrases and is explicitly invalid for quality gating.
