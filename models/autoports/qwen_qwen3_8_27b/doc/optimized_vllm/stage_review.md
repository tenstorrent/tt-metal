# Stage Review

Verdict: clean-pass

Independent final inspection of Stage 10, optimized-vLLM, for
`Qwen/Qwen3.8-27B`, on 2026-10-25. Reviewed live branch
`exp26-qwen38-bringup` at HEAD
`eacda58896fae0c866816fc2a5cba9ff5df554c3`, with uncommitted stage changes.
The standalone plugin is clean at
`d5ee19c6754905e7a9a69c79d9e2ee4b99b84295`.

No required implementation or validation work remains for the supplied stage
contract. This accepts the selected terminal-suppression implementation and its
recorded performance parity; it does not establish an end-to-end speedup or
general release readiness. The post-review local checkpoint and status receipts
remain the stage owner's next step; no push is authorized.

## Required Work

- None.

## Other Concerns

- Primary performance is effectively unchanged. Independently recomputed raw
  4096-input/128-output, one-request/concurrency-1, physical-B1 results are:

  | Metric | Before | Selected |
  |---|---:|---:|
  | Median TTFT, ms | 445.240526 | 447.392514 |
  | Mean TPOT, ms | 27.611422 | 27.621632 |
  | Median ITL, ms | 27.615973 | 27.613855 |
  | p99 ITL, ms | 28.345172 | 28.369549 |
  | Aggregate output tokens/s | 32.385304 | 32.357449 |
  | `1000 / mean TPOT`, tokens/s/user | 36.216896 | 36.203509 |

  Both complete exactly 4096 input and 128 output tokens with zero failures.
  Before/after server argv and recorded environments are identical. Only
  `generator.py` and `generator_vllm.py` differ in the runtime source manifests.
  Warmup is enabled in both server logs. The selected full-model control contains
  three 4096/128/B1 autoregressive, nonblocking, device-sampled records; its median
  decode rate is 36.170903 tokens/s/user. The reported ratio 1.000901 is correct,
  with the HTTP versus buffered-generator timing distinction disclosed.
- Secondary CI 100/100/32 completes all 32 requests before and after: mean TPOT
  112.060190 to 112.049889 ms, aggregate output 237.214519 to 237.238314 tokens/s.
  Its matched B32 manifest is valid. The separate one-request benchmark on a
  physical B32 server is correctly labeled and is not used as the B1 headline.
- The measured target-concurrency fixture is 32 conversations at 4130 tokens
  followed by 33-token appends. The configured 289-ID hybrid pool does not prove
  the planning workload of 32 simultaneous 32768-token conversations. The
  agentic ledger records the layout/pool shortfall and does not lower native
  context or call the configured pool a universal physical model limit.
- B8 acceptance is the tracked full sampling compatibility run, with its own
  scheduler budget 32768. It is explicitly ineligible as performance evidence.
  B16 uses budget 65536 and completes 16 concurrent 4096/128 requests; B32 uses
  budget 131072 and threshold 4096, with full mixed-traffic and CI acceptance.
  The batch configuration table preserves all four required capacities.
- Long eager prefills still pause decoding: the selected mixed fixture has a
  maximum SSE pause of 31.916428 seconds. This is disclosed and investigated,
  not presented as a low-latency service guarantee. The snapshot-batching trial
  reduced metadata uploads from 75 to 15 for the same 39 saves and 34 restores,
  but mixed time rose by 0.132016 seconds and B16 TTFT changed by only -5.497691 ms.
  Rejecting its extra implementation is supported by measurements.
- All 18 packaged/shared greedy and sampled outputs were read. Every one of
  the nine greedy outputs exactly matches its full-model control, with matching
  prompt text/rendering. Three packaged greedy completions reach the same
  256-token cap as their controls. Sampled story/thermodynamics and shared code
  also have incomplete final text at the cap. Raw reasoning remains in API
  content; the structured answer after `</think>` is correct JSON, while the
  entire response is not JSON-only. These are controlled smoke-test limits.

## Hard-Check Gaps

- Passing lifecycle status alone does not assess text quality. Raw synthetic
  completion fixtures include prompt bytes/replacement characters; comparison
  with prior serving and separate prompt-correct quality runs are necessary.
  Those controls exist and were checked here.
- The reduced terminal-suppression probe teacher-forces the first decode input
  after sampling. It proves exact terminal logits, sampled IDs, subsequent
  outputs and cache state under the compared chunk sequence, but is not alone
  a feedback proof. The separate adapter contract compares resident asynchronous
  feedback against authoritative synchronous submissions, uses deliberately
  invalid host tokens/positions on resident steps, crosses a changed used page,
  and checks exact tokens, positions and state. Existing all-layer automatic
  allocator-growth controls provide the complementary server coverage.
- The final B1 SIGTERM shutdown does not emit a graceful adapter-destructor
  counter audit. Source/manifest gating, selected same-source B32 event counters,
  and B16's shutdown audit establish the relevant no-host-fallback contract.
  The independent read-only process scan finds no live server/engine and no
  accessible Tenstorrent device descriptor; 211 inaccessible processes remain
  outside that descriptor scan's scope, as already disclosed in `cleanup.json`.
- No serving profiler, device time, roofline, broad maximum-context accuracy,
  long soak, or tail-population claim is established. These are explicit limits,
  not missing required new tests for this small serving change.

## Anomaly Ledger

- Observed anomaly: allocation warning mentions possible corruption while traces
  are live in tracking-disabled performance runs.
  Evidence: `after_b32_server.log`; selected tracked manifest and events.
  Affected path: eager prefill interleaved with retained decode/sampling traces.
  Control or comparison: same runtime hashes, tracking 1, program-cache skip 0,
  112 completed requests, stable captures, exact reduced controls.
  Likely subsystem: generic allocator diagnostic rather than a tracker violation.
  Investigation performed: checked source hashes, generator preparation ordering,
  absence of new acknowledgment/scope exemptions, raw counters and chunk events.
  Resolution: controlled for the exercised signatures.

- Observed anomaly: synthetic low-ID fixtures emit nonlinguistic text and prompt
  bytes in initial SSE chunks.
  Evidence: `after_b32_lifecycle.requests.json`.
  Affected path: synthetic raw completion lifecycle traffic.
  Control or comparison: exact equality against
  `readiness_vllm/full_b32_budget131072_lifecycle.requests.json` for first
  137/138/139, chunked 8195/12291, and prefix cold/hit 4130.
  Likely subsystem: fixture and completion/detokenization behavior.
  Investigation performed: independently compared all seven raw strings and
  reviewed all 18 separate chat-template outputs.
  Resolution: controlled; synthetic traffic is not language-quality evidence.

- Observed anomaly: long-prefill pauses persist despite terminal suppression.
  Evidence: `after_b32_lifecycle.json`, raw request timings and chunk events;
  rejected `final_b16_*` and `final_b32_*` measurements.
  Affected path: physical-B32 eager prefill and recurrent checkpoint handling.
  Control or comparison: fixed scheduler regime, prior lifecycle, measured
  snapshot-batching alternative, and restored selected source.
  Likely subsystem: prefill/admission work outside the steady decode loop;
  repeated snapshot dispatch was tested as one hypothesis.
  Investigation performed: re-derived mixed percentiles and maximum pause,
  checked concurrent continuation events, compared candidate latency and counts.
  Resolution: controlled limitation, with no gain or physical lower-bound claim.

- Observed anomaly: initial cache-path failure and pre-model mesh-open failure.
  Evidence: `probe_cache_environment_failure.log`, `probe_mesh_open_failure.log`,
  recovery logs and subsequent successful reduced/full-serving artifacts.
  Affected path: probe startup.
  Control or comparison: corrected cache environment, bounded list/reset/list,
  mesh smoke, then accepted tracked and full-model serving runs.
  Likely subsystem: environment and Ethernet initialization.
  Investigation performed: inspected retained failure/recovery evidence and
  successful follow-ups; no recovery commands were executed by this reviewer.
  Resolution: fixed/recovered by the stage owner.

- Observed anomaly: watcher ACTIVE_ETH image exceeds its code capacity.
  Evidence: `terminal_suppression_watcher.log` and
  `terminal_suppression_watcher_noeth.log`.
  Affected path: watcher mesh initialization.
  Control or comparison: compute watcher 10 with only ETH disabled passes the
  exact-state probe and allocation checks.
  Likely subsystem: instrumentation capacity.
  Investigation performed: checked the scoped retry and passing artifact.
  Resolution: controlled tooling limit; no Ethernet watcher coverage claim.

- Observed anomaly: two HTTP client close tasks log `Event loop is closed`.
  Evidence: `after_b8_sampling_tests.log`; inherited full B8 sampling log.
  Affected path: asyncio/httpx client teardown in a passing test.
  Control or comparison: 76 passed, 74 explicitly classified skips, zero failed;
  later assertions complete and final serving processes are gone.
  Likely subsystem: test-client lifecycle.
  Investigation performed: inspected exception context and final suite result.
  Resolution: controlled harness warning, not a model-test failure waiver.

- Observed anomaly: an earlier B1 process overlapped experimental source edits.
  Evidence: `superseded_b1_status.json`, excluded artifact index entries,
  selected manifest, and source-restoration receipt.
  Affected path: run provenance.
  Control or comparison: fresh `selected_b1_*` evidence; all accepted manifest
  hashes match the current source exactly.
  Likely subsystem: evidence collection during experimentation.
  Investigation performed: independently checked 32 indexed artifact hashes,
  every accepted manifest's runtime hashes, and the two actual source changes.
  Resolution: fixed by exclusion/restoration/rerun; rejected `final_*` evidence
  is not used as selected acceptance.

- Observed anomaly: API fingerprint suffix differs between B1 and B32 artifacts.
  Evidence: shared qualitative responses versus mixed lifecycle SSE responses.
  Affected path: provenance interpretation.
  Control or comparison: installed vLLM
  `entrypoints/serve/utils/fingerprint.py` derives the suffix from
  `vllm_config.compute_hash()`, not a source commit; server version is 0.30.0.
  Likely subsystem: expected configuration fingerprinting.
  Investigation performed: read the installed source without importing vLLM.
  Resolution: controlled; different capacity/configuration is not evidence of a
  dependency revision change.

- Observed anomaly: stage checker exits 1 and final shutdown force-kills an
  EngineCore before adapter destruction.
  Evidence: `stage_check.log`, `stage_check_verdict.json`, selected server log,
  `cleanup.json`, and independent process scan.
  Affected path: advisory classification and process teardown.
  Control or comparison: current native-262144 manifests; inherited measured
  weight copies; explicit planning-pool shortfall; no live resource owner.
  Likely subsystem: broad historical-log advisory checks and abort-timeout-0
  shutdown behavior.
  Investigation performed: checked actual advisory rows, final completion
  evidence, zombie classification, and accessible process descriptors.
  Resolution: controlled; no critical checker failure or live engine remains.

## Scope Inspected

- Goal/skill paths: supplied original Stage 10 contract, including exact
  4096/128/B1/concurrency-1 headline override; installed 0.1.21 `stage-review`,
  `model-bringup` startup, `vllm-integration`, `optimize`, `agentic-serving`,
  `tt-device-usage`, `tt-enable-tracing`, and `qualitative-check`; checkout
  `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`.
- Artifacts: Stage 10 README/work log/checklist/topology/anomaly ledger; source
  restoration and artifact index; all accepted B1/B8/B16/B32 command manifests;
  primary and CI raw/normalized before/after metrics; rejected candidate metrics;
  terminal and adapter correctness artifacts; sampling/host/watcher/server logs;
  lifecycle events/raw requests; selected packaged/shared outputs and actual
  full-model controls; context and agentic ledgers; prior automatic page-growth
  controls; selected datatype-sweep B1 autoregressive records; stage check,
  cleanup, and local telemetry packet with adjacent exact template.
- Code: complete Stage 10 runtime diff; adapter prefill grouping/suppression,
  cache ownership, fixed slots, async decode/read/finalization and teardown;
  generator preparation, persistent inputs, prefill continuation, terminal,
  sampling and decode; terminal/adapter probes, lifecycle runner, serving shell
  runner and host tests. Plugin was inspected as unchanged context.
- Commands run: read-only `git status/diff/rev-parse`, `rg`, `cat`, `sed`,
  standard-library JSON/hash/string/arithmetic analyses; container reads of
  installed fingerprint source and `/proc` process/descriptors. No TTNN import,
  hardware access, test execution, server launch, reset, profiler or device
  opening. The sole write is this requested report.

Independently established runtime/evidence properties:

- Generator SHA256
  `91c8ad90268bcbb42185e9c3ad4cb143d798917316c93ba7b464410812ccacbb`;
  adapter SHA256
  `315bd0fe2ac27fb0c9cbfc6efa655664ddee53bfb8c880d0c10d9b9f2b746307`.
  Every source entry in the selected B1 and accepted B8/B16/B32 manifests
  matches the present checkout. All 32 artifact-index checksums match.
- Terminal suppression is conditional only on an intermediate physical piece
  or a scheduler-suppressed output. Decoder/cache work still executes; final
  emitted pieces retain terminal and canonical sampling. `return_all_logits`
  rejects incompatible suppression. Off-grid bridge trace work is unchanged.
  Preparation warms the terminal-capable superset before capture; no new
  persistent buffer, trace variant or allocation exemption is introduced.
- The selected tracked lifecycle contains 904 events from one PID: one prepared
  baseline, 672 decode submissions, 112 releases, 30 prefill submissions and
  completions, and 59 pieces (35 suppressed, 24 terminal-bearing). Captures
  remain 4, retirements 0 and host-logit reads 0 throughout. Actual continuation
  starts include 4096/8192/12288 with partial 3-token tails while another slot
  decodes. All 112 raw requests complete with exact prompt/output counts.
- Raw mixed TTFT p50/p95/p99 reproduce 4972.054708/32628.207678/32629.142009 ms;
  SSE pause p99/max reproduce 426.075466/31916.427664 ms. All 32 second-turn
  admissions restore at 4096, giving 131072/132160 = 99.176755% reuse.
- Device sampling and nonblocking replay are preserved. The async method returns
  device output, reads one replicated token tensor asynchronously, and leaves
  synchronization to the plugin before formatting. Logs demonstrate overlap.
  The current adapter probe has exact token/position/cache equality, zero steady
  token/position refreshes, one intentional changed-page refresh, and no extra
  read at async host finalization. Existing full-model page-growth evidence
  covers actual allocator updates at 4096 and 8192 with three concurrent requests
  and six exact comparisons to non-overlapped control; the changed code does not
  alter that decode/plugin path.
- `head_bfp8_hifi2`, all 64 layers, real adapter/plugin registration and
  max-model-len 262144 appear in accepted runtime logs/configuration. No
  selected performance manifest enables host compatibility or device profiling.
  B16's audit and tracked B32 events have zero host-logit reads.
- Telemetry packet `3afcc9f8-8d57-46ac-b083-eb90c11eeb67.json` matches its
  adjacent template's schema/run/stage/attempt/kind/workload identifiers.
  Primary metrics match raw evidence; device-time and roofline fields are null
  with the no-profiler reason. Review/commit notes are appropriately pending
  until this review and the subsequent local checkpoint are recorded.

## Residual Risk

- This verdict applies to the stated source hashes and selected artifacts.
  Implementation changes after review invalidate affected acceptance evidence.
- Native-context capacity and precision selection are inherited from unchanged
  earlier stages; current serving tests do not claim a new exhaustive
  maximum-context accuracy evaluation or every batch/context combination.
- The planning long-context pool shortfall, capped qualitative replies, raw
  reasoning output, host sampling compatibility for unsupported conformance
  parameters, and long mixed-traffic pauses remain explicit product limits.
- Single primary measurements support parity, not a statistically established
  speedup or a tail-latency population. No extra broad sweep, profiler or soak
  is required to close this inspected change.
