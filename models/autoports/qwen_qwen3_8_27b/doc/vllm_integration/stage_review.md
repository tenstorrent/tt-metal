# Stage Review

Verdict: clean-pass

Independent final review of Stage 09, Qwen/Qwen3.8-27B, on 2026-10-25. Live
tt-metal tree based on `fe82645a041cf665a621dc6211c0e9354edb995b`, with the
standalone plugin based on `1badbd02bb6911dfdb59c206bdc00c6c1e8dc0bf` and local
changes in both repositories. Initial incomplete-evidence findings are closed
by the final artifacts and documentation corrections inspected below. Stage-owned
local checkpoint commits follow this clean-pass; no push is authorized.

## Required Work

None. The final-source B1 measurement, final B32 lifecycle and CI benchmark,
per-batch configurations, sampling/quality controls, cleanup record and compact
telemetry packet satisfy the supplied Stage 09 contract. No additional
implementation fix or hardware rerun is required by this review.

## Other Concerns

- Full sampling acceptance is an explicit compatibility-mode result: 76 passed,
  74 skipped, zero failures. Seeded requests use the plugin's host policy for
  mixed-feature reproducibility. This must remain distinct from all-device
  sampling and performance claims. The separate device-only route controls
  reproduce across three request orderings with zero host-logit-read delta.
- The initial B32 admission anomaly is investigated and improved by the final
  scheduler configuration. Cold 32-request phase duration falls from 524.390 to
  33.763 seconds and prefill submissions from 64 to four; whole-workload TTFT P99
  falls from 521623.397 to 32686.979 ms. Maximum SSE pause remains 31959.208 ms,
  so the much lower P99 pause of 425.512 ms does not establish low worst-case
  latency. These are tracking-enabled synthetic lifecycle timings, separate
  from the tracking-disabled serving benchmark.
- Native context remains 262144. The 32-conversation reuse measurement is
  4130 initial tokens plus 33 appended tokens per conversation, not a measured
  32 by 32768 working set. The documented pool shortfall must remain explicit.

## Hard-Check Gaps

- Three shared chunked-prefill passphrase-recall tests were skipped because the
  shared runner did not supply their opt-in budget. The custom lifecycle proves
  actual scheduler continuations, trace retention, completion, and cold/hit
  prefix equality using synthetic tokens; it is not the same semantic recall
  test. Reduced continuation probes additionally compare logits and every state
  field across awkward splits. This is a coverage distinction, not a demand for
  another full suite or long-context sweep.
- Device allocator measurements are preparation checkpoints, not peak or
  per-request memory measurements. The lifecycle records scheduler cache metrics,
  and source inspection finds fixed pools, preallocated bucket/state buffers,
  and temporary eager outputs without a growing request cache. These establish
  bounded ownership more directly than successful HTTP responses alone, but do
  not measure allocator high-water behavior for every advertised prompt length.
- The original B32 manifest lacks the later detailed environment/source hashes;
  final B32 lifecycle and performance manifests supply them and match inspected
  implementation files.
- B16 shutdown logs show abort timeout zero and forced EngineCore termination.
  The adapter printed its audit before trace release, but its final
  `shutdown_release` event is absent. The owner's subsequent empty-process audit
  can establish process cleanup; it must not be described as completed graceful
  trace/mesh teardown.

## Anomaly Ledger

- Observed anomaly: a client cleanup exception appears in the passing full
  sampling log.
  Evidence: `readiness_vllm/full_b8_sampling_tests.log:186`,
  `AsyncClient.aclose`, `RuntimeError: Event loop is closed`.
  Affected path: pytest client HTTP connection teardown.
  Control or comparison: the test and all 76 executed cases passed; the traceback
  is entirely within HTTP client/event-loop cleanup, not the server adapter.
  Likely subsystem: asynchronous client resource cleanup across pytest loops.
  Investigation performed: inspected the complete captured traceback and server
  overlap logs; no model/device failure is established by this exception.
  Resolution: controlled for stage correctness; retain as a client cleanup issue.

- Observed anomaly: thinking text and incomplete story/thermodynamics answers.
  Evidence: all twelve outputs in `vllm_qualitative_outputs.json`, and the nine
  outputs/rendered prompts in `full_model_controls.json`, were read directly.
  Affected path: chat generation with a 256-token cap.
  Control or comparison: the checkpoint template opens a thinking section;
  standalone controls use the same rendered format and truncate the same topics.
  Likely subsystem: checkpoint output format and requested token budget.
  Investigation performed: checked coherence, topic, repetition, gibberish,
  language, and visible contamination. French appears in the requested translation;
  Fibonacci code and the completed explanations are coherent and relevant.
  Resolution: controlled; these are not claims of complete answers at this cap.

- Observed anomaly: mixed host/device seeded sampling previously differed across
  request orderings.
  Evidence: `sampling_route_controls.json`, `full_b32_initial_sampling_tests.log`,
  and `doc/vllm_integration/generator_autofix.md`.
  Affected path: explicitly enabled compatibility sampling with penalties.
  Control or comparison: three device-only orderings agree; two host-logprob
  orderings agree; final compatibility profile passes with seeded-host policy.
  Likely subsystem: transition between distinct host and device RNG streams.
  Investigation performed: inspected routing, seed reset/subset preservation,
  plugin reload planning and raw control outputs.
  Resolution: controlled with an explicit compatibility policy; cross-route RNG
  identity is not claimed.

- Observed anomaly: generic unsafe-allocation warning in the tracking-disabled
  final B32 performance log at line 182 and final B1 log at line 287.
  Evidence: `full_b32_performance_server.log`, `final_b1_server.log`, reduced
  B1 contract proof, and the matching final lifecycle
  command/source manifest with allocation tracking enabled and program-cache
  exclusion zero.
  Affected path: post-capture allocation in performance mode.
  Control or comparison: same implementation and scheduler/cache configuration
  complete all 112 lifecycle requests and interleaved trace variants under the
  allocation tracker; no unsafe-allocation exception or trace retirement occurs.
  Likely subsystem: the allocator's generic warning when tracking is disabled.
  Investigation performed: compared manifests and inspected tracker-enabled
  event slices and code allocation lifetimes.
  Resolution: controlled by representative tracking-enabled acceptance evidence;
  the warning is not proof of detected corruption.

- Observed anomaly: stale pending/synchronous fields survived in otherwise final
  batch and agentic-serving documents.
  Evidence: the B8 row initially referenced its synchronous control as the active
  command and marked its async comparison pending; agentic-serving status fields
  still called measured benchmarks incomplete and omitted tested batch 16.
  Affected path: evidence attribution and stage reporting.
  Control or comparison: source manifests, completed raw benchmark outputs,
  B8 growth comparisons and B16 request counts.
  Likely subsystem: documentation reconciliation.
  Investigation performed: reported exact fields, then reread the corrected JSON.
  B8 now identifies the async command/events with the sync run retained as a
  control; active agentic status reflects measured benchmarks and batches
  1/8/16/32. Historical entries remain explicitly historical.
  Resolution: fixed without runtime changes.

## Scope Inspected

- Goal/skills: supplied Stage 09 contract, including the explicit
  4096/128/B1/concurrency-1 headline override; installed version 0.1.21
  `stage-review`, `vllm-integration`, `agentic-serving`, `qualitative-check`,
  `tt-enable-tracing`, `tt-device-usage`, and model-bringup startup instructions;
  checkout `TraceCorrectness.md` and enabled plugin inventory.
- Documents: evidence index, README, work log, AUTOFIX, runtime audit, serving
  stack, batch configurations, generator repair and preparation descriptions,
  context and selected-precision contracts.
- Code: complete adapter; generator preparation, prefill/continuation, sampling,
  replay, resident-input and state-remap paths; recurrent snapshot pool and copy
  kernel; hybrid cache specs; relevant model/precision wiring; modified plugin
  files, new `prefill.py`, and pinned asynchronous reload/submission implementation.
- Raw evidence: B8 asynchronous/synchronous page-growth requests and event slices;
  B32 lifecycle requests and event slice; sampling log/verdict and raw route
  controls; standalone logit controls; qualitative outputs and controls; reduced
  generator/adapter/state-pool/continuation probes; archived B1 benchmark/raw JSON
  and source manifests.
- Independent re-derivation: all six async outputs equal their control token arrays
  for all 4160 tokens, with matching prompt hashes. All inspected lifecycle/growth
  event slices retain a single PID, four captures and zero retirements. All 112
  lifecycle requests completed with exact input/output counts. Cold/hit prefix
  output matches. All 24 standalone/serving top-20 distribution comparisons have
  identical token sets and maximum absolute logprob difference zero. The B8
  manifest's implementation hashes match the inspected tree. Its preparation
  precision object equals the selected policy exactly, including every group,
  fidelity, CCL/KV dtype and empty exception map.
- Subsequently supplied B16 benchmark, raw JSON, manifest, events and server log
  were also checked: 16/16 requests completed at 129 input/8 output tokens each,
  concurrency 16, with 2064 total input and 128 total output tokens. Source
  hashes match current code, tracking/program-cache settings are 1/0, and host
  compatibility is absent. Workload events retain four captures, zero
  retirements and zero host-logit reads. This is a tracking-enabled configuration
  validation row, not the headline performance result.
- Final B32 follow-up: inspected `full_b32_budget131072_*` lifecycle, raw requests,
  source/environment manifest and host timeline, plus
  `full_b32_performance_*` archives and canonical CI JSON. All 112 lifecycle
  requests have exact counts; event slice 1:904 contains 903 events from PID
  1654081, four captures and zero retirements. Both masked/unmasked decode traces
  execute, and scheduler continuation starts 4096/8192/12288 coexist with decoding.
  All 111 greedy outputs are byte-identical to the earlier scheduler workload.
  Cached-prefix output and repeated greedy-after-sampled output agree. All 32
  repeated conversations restore at 4096, reusing 131072/132160 previous tokens.
  Both final source manifests match current code. The CI run completes 32/32
  requests, 3200 input/3200 output tokens, and 237.214519 output tokens/s; raw
  observed peak concurrency is 32. Its null `max_concurrency` means an unbounded
  burst of 32 total requests, not absent observed concurrency. Raw CI and archived
  copies agree, including TTFT P50 2395.195 ms, mean TPOT 112.060 ms, and ITL
  P50/P99 109.197/204.297 ms. Performance tracking, event logging and host
  compatibility are disabled as required.
- Final B1 follow-up: `final_b1_vllm_result.json` records one completed request,
  zero failures, exactly 4096 input and 128 output tokens, and concurrency one.
  Raw/normalized canonical copies equal the final archives. TTFT is
  445.545611903 ms, mean TPOT 27.591065716 ms, ITL P50/P99
  27.616490843/28.495862638 ms, and aggregate output throughput
  32.404760139 tokens/s. Independently recomputed `1000 / mean_tpot_ms` is
  36.243616332 tokens/s/user. The CLI explicitly sets 4096/128/1/1 and temperature
  zero. All 44 implementation hashes match the inspected tree; tracking,
  profiler, watcher, event logging, reduced layers and host compatibility are
  disabled/unset as appropriate. API and EngineCore logs resolve the actual
  autoport class and native context. The shutdown audit records all 64 layers,
  zero host-logit reads, 127 model and 128 sampling replays, one token refresh,
  one position refresh, two page-table refreshes total, and three B1 captures.
- Final reporting: README, work log, evidence index, stack, batch configurations
  and active agentic-serving fields agree with measured artifacts. Teacher
  forcing is explicitly a lower-bound reference at 203/100/B1, separately from
  the matching Stage08 autoregressive 4096/128/B1 result of 36.170902880 tokens/s.
  The final vLLM result does not show material decode overhead relative to that
  token-out baseline; this is not a claim that all service latency is optimal.
- Telemetry: packet
  `4362e94e-730c-4b3c-840b-f8f182e1bca1.json` under the supplied persistent run's
  `telemetry/packets/` directory matches its template's schema version, run ID,
  stage, attempt ID and kind. Workload is 4096/128/B1/concurrency one, source is
  `vllm`, and TTFT/decode values exactly equal the final raw-derived measurements.
  Missing device/roofline and dataset-accuracy quantities carry explicit reasons;
  no host-wall or synthetic metric is substituted for them.
- Cleanup: inspected `final_cleanup.json` and independently ran a read-only
  container `/proc` process/descriptor scan. No live serving process or
  accessible TT device descriptor was found. The same PID 1635300 is a zombie
  without device ownership. The independent scan encountered 202 inaccessible
  descriptor directories; the owner's host scan separately discloses 670
  inaccessible process directories. This is scoped process/ownership evidence,
  not universal visibility into unrelated users or proof of graceful teardown.
- Lifetime/ownership findings: generator-wide preparation precedes capture;
  request paths retain trace handles and backing buffers; no allocation
  acknowledgments or broad corruptible scopes were added. Sampling writes the
  next token into persistent `tokens`, positions advance in the decode trace,
  page-only changes preserve both, and plugin transitions invalidate residency
  and drain pending work. Recurrent snapshots have separate scheduler-indexed
  storage and exact copy primitives. B8 tracking is enabled before import with
  program-cache tracking retained; performance tracking is separately disabled.
- Commands run: read-only `cat`, `sed`, `rg`, `git diff/status`, short Python
  standard-library JSON/hash/comparison scripts, and read-only container `/proc`
  inspection. No TTNN import, hardware access,
  server launch, HTTP workload, reset, or test execution. Only this report was
  written.

## Residual Risk

This review covers the final measured implementation in a live tree before local
checkpoint commits. It is not a full-capacity soak, exhaustive API conformance
audit, or independent execution of the hardware probes. The documented 32-second
long-prefill pause, unmeasured 32x32768 planning workload, compatibility sampling
limits, preparation-only memory checkpoints and process-visibility limits remain
explicit. None contradicts the supplied Stage 09 acceptance contract. No concrete
correctness defect or unsupported completion claim remains in the inspected scope.
