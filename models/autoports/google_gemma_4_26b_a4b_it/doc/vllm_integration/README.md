# Gemma 4 vLLM integration

Primary single-user **TTFT 2549.30 ms; decode 50.73 tokens/s/user**
(4096 input tokens, 128 generated tokens, batch 1, one concurrent request; greedy on device).
Full 30-layer model, selected datatype policy, async scheduling and traced decode.
Final shared runner succeeded: full sampling **72 passed, 1 skipped**, zero failures.
The all-vocabulary logprob test takes its documented default-cap skip; ordinary
top-N logprobs pass. Server shutdown and process cleanup passed. Independent
`stage_review.md` verdict: **clean-pass**, no required work. Local checkpoint
SHAs are recorded in `work_log.md`; nothing was pushed.
All 12 final shared qualitative outputs were read; no degeneracy or visible request contamination.

| Workload | TTFT P50 / P99 (ms) | TPOT mean / P50 / P99 (ms) | ITL P50 / P99 (ms) | Aggregate output tokens/s | 1000 / mean TPOT (tokens/s/user) |
| --- | --- | --- | --- | --- | --- |
| Primary: 4096 input / 128 output, B1 C1, 1 request | 2549.301 / 2549.301 | 19.713 / 19.713 / 19.713 | 19.619 / 27.752 | 25.331 | 50.728 |
| Secondary CI: 100 input / 100 output, 32-request burst, no client concurrency cap | 30815.582 / 30816.835 | 680.950 / 677.478 / 754.133 | 677.253 / 692.437 | 32.690 | 1.469 |

Primary completed 1/1 request with 4096 input and 128 output tokens. Its request
percentiles describe one request, not a population distribution. Secondary CI
completed 32/32 requests with 3200 total input and 3200 total output tokens;
server logs show 32 requests running together. Burst admission and prefill affect
CI TPOT, so the CI value is not the headline decode rate. Both profiles use
random token inputs, temperature 0, ignore_eos, and explicit generation config.

Normalized/raw artifacts: `../../readiness_vllm/vllm_benchmark.json` and
`vllm_result.json` (primary); `vllm_ci_serving_benchmark.json` and
`vllm_ci_serving_result.json` (secondary). Each normalized JSON contains the exact
benchmark command and workload. Corresponding `.log` files preserve execution.

Server and checks are launched through the shared readiness runner:
`full_server_command.json` and `final_checks_command.json` contain exact argv.
`runtime_environment.json` records runtime paths. Served max_model_len=262144
matches `../context_contract.json`; max_num_seqs=32, TP4/DP1, P300x2 (four
Blackhole devices), FABRIC_1D, trace_region_size=1000000000, trace_mode=decode_only,
sample_on_device_mode=all. Chunked prefill and prefix caching are disabled.
Environment: OMP_NUM_THREADS=8, writable TT_METAL_LOGS_PATH and VLLM_CACHE_ROOT;
the optional GEMMA4_AUTOPORT_ALLOW_HOST_SAMPLING=1 is enabled only to permit
shared constrained/logprob tests. Ordinary generation and both benchmarks use
canonical traced device sampling. No serving profiler was run.

The adapter delegates prefill, split sampling and decode to `tt/generator.py`.
Selected `head4_inner_all4_shared_down4` policy supplies the weight groups,
per-layer activation/CCL dtype groups and exceptions, BFP8 KV cache, and compute
fidelities. Construction validates the selected precision config; the generator
exposes its validated `precision_summary()` for inspection.
The plugin registers `TTAutoportGemma4ForCausalLM` in `register_tt_models()`.
vLLM owns hybrid sliding/full-attention cache pools. Per-layer page tables and
geometry-preserving aliases pass that storage to the low-level generator.

Contract evidence under `../../readiness_vllm/`:

- `adapter_changed_pages.json`: stale token/current-position inputs, canonical
  device feedback, deferred readback; one changed layer table refreshes once
  without recapture, unchanged tables do not refresh.
- `adapter_trimmed_batch.json`: trailing scheduler wire padding removed while
  preserving active/interior rows and external cache ownership.
- `requests_full_final.json`: nine 96-token generations with non-aligned
  31/63/95-token prompts isolated and concurrent, plus 33/1057/33 prompt reuse.
- `seed_lifecycle_final.json` and `seed_lifecycle_final_seed_only.json`: seeded
  targets exactly reproduce isolated controls when shorter peers finish, in both
  submission orders, with and without penalties.
- `resumed_prefill_reduced.json`: retained-output seed/history restoration passes
  a canonical same-logits device oracle; CPU tests additionally cover scheduler
  reconciliation of discarded async tokens before resumed prefill.
- `logit_determinism_full.json`: 22 exact full-vocabulary standalone/adapter
  comparisons across repeated and reordered rows, prefill plus three decode steps.
  `logprobs_full_server.json` and `logprobs_standalone_comparison.json`: live
  first-token top-20 logprobs reproduce and exactly match standalone log-softmax.

Reduced-layer artifacts are contract diagnostics, not full-model performance
or quality evidence. Full split-sampling trace allocation tracking also passes
the four 256-token controls in `qualitative_256_controls.json`.

`qualitative_verdict.md` records the final six greedy and six sampled responses,
pinned chat-template formatting, HF/selected-policy controls and limitations.
All six greedy prefixes match selected-policy controls; two extended greedy
outputs match standalone exactly. Awkward hyphenation and “own-contained” also
occur outside serving. Some 256-token responses are incomplete. This is a serving
regression smoke, not a scientific accuracy certification. `degeneracy.json`
has no findings. Final artifact hashes are in `qualitative_control_comparison.json`.

The inherited stochastic device sampler caps effective top-k at 32. Explicit
optional host compatibility serves constrained/logprob requests using vLLM CPU
semantics; it does not replace the benchmark path. Greedy uses the canonical
split sampler, with no adapter argmax or host token-feedback loop. Runtime mode
selection, warnings, trace lifecycle and teacher-forcing comparison are audited
in `runtime_audit.md`.

The teacher-forcing control is only a decoder latency lower bound at its own
161-input/100-output/B1 workload, not a serving performance result. Removing
unused padded rows resolved measured vLLM overhead at the primary workload;
`baseline_padded_decode/` preserves the same-harness baseline. Stable decode
does not upload host token/position/history state or unchanged page tables.

Sampling history and evidence-backed fixes are in `work_log.md` and the
`AUTOFIX_*.md` reports. The final full profile and cleanup are recorded in `../../readiness_vllm/final_validation.json`
and `final_cleanup.json`. The independent `stage_review.md` verdict and local
checkpoint SHAs are recorded at closure.
