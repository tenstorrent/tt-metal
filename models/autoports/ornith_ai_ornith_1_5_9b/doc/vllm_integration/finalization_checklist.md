# Final 64-candidate evidence checklist

This is a reporting and promotion checklist, not an acceptance result. It was
prepared by CPU-only source/artifact inspection on 2026-11-13. Paths below are
relative to this model directory unless stated otherwise. The root agent owns
device scheduling, execution, review, promotion and commits.

## Preserve the evidence boundary

- [ ] Record the final main-checkout commit/diff, common sampling implementation,
  native sampling kernel sources, loaded TTNN module/build identity, plugin HEAD,
  upstream vLLM version, HF revision and precision selection. Both final drivers
  now invoke `snapshot_native_sampling.py` and require
  `SAMPLING64_REBUILD_PROVENANCE`. Inspect their `loaded_source` record (or
  `loaded_model_source` for the tracked driver): `native_sampling_provenance.json`,
  copied `native_main_rebuild_provenance.json` and `native_sampling/` sources.
  The helper validates the exact source set against the five-step successful
  rebuild and records actual main TTNN module/library mappings and stat identity
  in its importing process. It copies no binaries and does not rehash them.
  This is explicitly import-process evidence, not a claim to have inspected the
  server worker's mappings. Correlate the launch environment/source snapshot and
  loaded serving capability; an API docstring alone is not implementation proof.
- [ ] Verify canonical construction uses capacity64, common `ttnn.sampling` receives
  that capacity, and adapter `max_device_top_k` agrees. Preserve default32 behavior
  for other callers. Preserve the existing 32-capacity reference artifact verbatim.
- [ ] Record which evidence predates integration. Current root readiness artifacts,
  `corrected_tracked/`, and the README's 67-pass/75-skip and mixed-trace counts are
  prior32 evidence. Do not rename those results as final64 results.
- [ ] Keep old artifacts and failed experiments intact. The degenerate-output gate
  recursively discovers **all** `vllm_qualitative_outputs.json` files, including
  archive directories; the context gate recursively scans `readiness_vllm/` and
  `doc/vllm_integration/`. An archive is not excluded from checks. Resolve any real
  finding and explain diagnostic scope; do not silently delete or alter evidence.

## Source-specific validation after integration

- [ ] Retain native64 primitive tests and default32 comparison, covering two-tile
  addressing, width128, local ranks33–64, stable ties/zero values, seeded replay and
  supported architectures actually tested. Separate host/build checks from device
  results; do not imply untested architecture coverage.
- [ ] Run `tests.vllm_sampling_capacity_probe --expected-capacity 64
  --compare-reference "$MODEL_DIR/readiness_vllm/sampling_capacity32.json"
  --output "$MODEL_DIR/readiness_vllm/sampling_capacity64.json"` through the model
  module prefix `models.autoports.ornith_ai_ornith_1_5_9b`. Require TP4 global IDs,
  real primitive capacity64 and gathered width256, k40/k64 tail draws, exact
  eager/trace/resume sequences, seeds and penalty histories, and unchanged unique
  and supported-tie greedy results against32. The isolated warmed k1/k20/k40 trace
  timings are auxiliary; tracking-on numbers are not serving performance.
- [ ] Re-run the actual adapter adversarial battery on the final source:
  `python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.vllm_adapter_decode_probe
  --output "$MODEL_DIR/doc/vllm_integration/adapter_decode_sampling64.json"`.
  Require stale token/current-position objects untouched in resident steps,
  pending submissions and independent outputs, unchanged-table steady steps,
  page-only refresh without token/position reset, inactive peers, remapping and
  slot-reset ownership. Existing adapter JSONs do not establish final64 coverage.
- [ ] Re-run `tests.vllm_seed_decode_probe` and
  `tests.vllm_original_boundary_probe` with distinct final64 `--output` paths.
  Their current top-k32 fixtures still exercise the new capacity64 pipeline and
  adapter seed/history transitions; the capacity probe supplies k40/k64 coverage.
  If diagnostic op keys need preparation under allocation tracking, warm them
  before capture; do not disable tracking or exempt those allocations.
- [ ] Keep `readiness_vllm/prefill_horizon_device.json` as the already-passing
  intermediate-discard/horizon regression (unknown512 to known4096, missing129
  tail to known200). Re-run only if subsequent edits affect that path. Its status
  is already `pass`, and the README now records that completed validation.
- [ ] Re-run relevant host contract/seed/replay tests and targeted pre-commit after
  integration. The new primitive must not change canonical cache ownership,
  split sampling, finite discarded-row placeholders or final host compatibility.

## Full-model collection and checks

- [ ] Execute `doc/vllm_integration/run_sampling64_tracked_acceptance.sh` under the
  device owner's serialized schedule. Require successful full sampling profile,
  fresh qualitative texts, `mixed_trace.json`, `lifecycle_trace.json`, targeted
  `chunked_prefill_tests.log`, `logit_http.json`, P2 route proof, P3, P5 accuracy,
  `runtime_audit.json` and normal cleanup in `readiness_vllm/sampling64_tracked/`.
  Its `acceptance_index.json` deliberately remains pending text/oracle review.
- [ ] Inspect mixed/lifecycle events from that same EngineCore: non-aligned
  requests, actual continuation starts and final tails, concurrent decode, changed
  row order and sampling modes, baseline trace IDs/signatures, and zero additional
  captures/retirements/evictions. Map allocator-driven page growth across multiple
  boundaries to distinct requests and the trustworthy control. The reduced adapter
  battery's page-only case swaps **unused future pages**; it is not this proof.
  Prior32 functional evidence is `corrected_tracked/mixed_trace.json` plus
  `lifecycle_trace.json`: concurrent distinct lengths129/130/131 (then137/138/139)
  generate70 tokens, crossing consumed positions160 and192, and match their
  explicit-logprobs non-overlapped controls exactly. Each workload records22
  device page-only refreshes,248 unchanged-table resident steps and69 host steps
  with input reload. Pinned `async_decode.py` excludes logprobs from steady
  overlap and drains pending steps before such builds. Do not substitute the
  earlier diagnostic4/7 all-space reduced-model control for this proof. Repeat
  this functional coverage on final64; current events do not expose a per-request
  physical-page ledger, so do not claim that additional ledger-level proof.
- [ ] Run the matching standalone B32 oracle using `tests/vllm_logit_determinism_probe.py
  tt --http-result "$MODEL_DIR/readiness_vllm/sampling64_tracked/logit_http.json"
  --output <fresh-final64-result>`. Do not compare final64 HTTP against an older
  HTTP artifact. Retain greedy/replay/model-state evidence needed to interpret
  numerical and request-contamination results.
- [ ] Read every new qualitative output and record coherence, topic, repetition,
  gibberish, language drift and cross-request contamination. Associate exact prompt
  formats and retained HF/TT controls; explain any reuse of older controls. A
  synthetic greedy match alone does not establish full-model text quality.
- [ ] Execute `run_final_measurements.sh final64` after tracked acceptance: separate
  tracking-off B32 and B1 servers, no profiler/watcher, same final sources/config.
  Require P2 all default profiles at users1,32 and load32, primary B1 and B32
  benchmarks, CI burst and prefill burst. The final script reads max-device-top-k
  from the adapter automatically; installed P2 resolves a literal64 or same-file
  constant. Its missing explicit `--max-device-top-k` is not itself a defect.
- [ ] Review P2 raw profiles/ratios and route evidence, P3 critical checks and any
  explicit advisory, and all nine P5 accuracy length/depth cases. Do not promote
  a tracked reasoning-only P2 result over the final untracked full-profile P2.
- [ ] Audit fallback separately: deliberate host-logprobs/shared-test compatibility
  is explicit; final greedy and supported sampled device performance paths have
  no host sampling, full-logits readback or host token feedback. Record allocation
  tracking, sampling width, async flag and capture counts from the loaded source.
- [ ] Retain `context_acceptance.json` (262143 input + 1 output) and context262144
  contract; no capacity rerun is required solely by candidate width. Check final
  launch commands still serve262144. Confirm every owned server/EngineCore exits
  normally and final holder/process audit shows devices released.

## Explicit artifact promotion

Archive current canonical files before promotion, copy exact outputs, and keep an
index mapping each promoted file to its run, source snapshot, launch summary and
raw data. Never manufacture a combined runner success or overwrite raw paths.

| Canonical artifact | Final source / rule |
| --- | --- |
| `readiness_vllm/sampling_tests.log`, `full_sampling_runner.log` | `sampling64_tracked/`, full profile; retain exact pytest counts and skips |
| `readiness_vllm/vllm_qualitative_outputs.json` | Fresh `sampling64_tracked/` output after manual review |
| `readiness_vllm/qualitative_observations.json` and parity/control references | Fresh review of those exact outputs; older control reuse explicitly justified |
| `readiness_vllm/mixed_trace.json`, lifecycle/runtime audit artifacts | Fresh tracked64 results; retain complete event log/signature mapping |
| `readiness_vllm/vllm_benchmark.json` | `final64_b1_untracked/`: 128 input / 128 output / 1 request / concurrency1, B1 capacity |
| `readiness_vllm/vllm_ci_serving_benchmark.json` | `final64_b32_untracked/`: 100 / 100 / 32, default unlimited concurrency, B32 capacity |
| `readiness_vllm/vllm_prefill_burst_benchmark.json` | `final64_b32_untracked/`: 128 / 32 / 32, concurrency32, serial prefill |
| `readiness_vllm/vllm_run_summary.json` | Preserve an actual final B32 launch summary with command, block32 and context262144; link the separate B1 launch summary beside the promoted headline benchmark |
| `readiness_vllm/prefix_cache_check.json` | Keep explicit `recurrent_state` exclusion consistent with serving contract and final prefix setting; no fictitious hit test or batched/serial ratio |
| `doc/eval_readiness/harness_requests.json` and `harness_requests/` | Final untracked P2 writes these canonical files directly; retain matching raw `runs.json` and source/run attribution |
| `doc/eval_readiness/chat_conformance.json` plus raw directory | Promote matching files from `sampling64_tracked/eval_model/doc/eval_readiness/` |
| `doc/eval_readiness/long_context_accuracy.json` plus raw directory | Promote matching files from the same tracked64 eval directory |
| `doc/context_contract.json`, `doc/full_model/serving_contract.json` | Keep accurate capability/cache/prefix declarations and exact evidence references |

## Report, gate, review, commit

- [ ] Put primary **B1 128/128/1 concurrency1** TTFT and `1000/mean_tpot_ms`
  tokens/s/user at the top of README. Next report **B32 CI 100/100/32** as secondary;
  every number carries its workload/capacity. Record TTFT/TPOT/ITL p50 and p99,
  output throughput, sampling profile, TT config, context, commands and raw JSON
  paths. Keep B32 single-user and prefill burst separate. Compare full-model
  teacher-forcing only as a lower bound, with its different workload disclosed.
- [ ] Update README and work_log: final64 capability/optional host boundary,
  selected precision, plugin registration provenance, fresh counts and qualitative
  verdict, P2/P3/P5 results/advisories, actual process status and exact artifacts.
  Remove obsolete pending statements only when their fresh evidence is complete.
- [ ] Run the installed `prompts/model_bringup_multigoal/09-vllm.check.sh` with the
  model environment. Its exact checks are degenerate output, context contract,
  serving contract, prefill features, and eval readiness. Exit2 is a critical
  failure; exit3 is checker error. Explain exit1 advisories. This script does not
  by itself validate every original-goal sampling/async/performance requirement.
- [ ] Invoke final independent stage-review only after this evidence is reviewable;
  fix/re-review findings to `clean-pass` (or permitted `pass-with-carried` after
  three rounds). Locally commit all stage-owned touched-repo changes, log exact
  SHAs and do not push. Complete the goal only after all requirements are met.
