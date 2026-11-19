# Stage Review

Verdict: clean-pass

Independent round 2, diff-only follow-up of stage 9, Qwen/Qwen3.6-27B,
on 2026-11-17. Reviewed committed snapshot
`450c36d7c4864f69f2e8365b7c62e4ff3506e495` against round 1 baseline
`b527d5aafa7ea4e12a9eca2dfdd350ff19dfed74` and its documentation-closeout
appendix. The sole required finding is closed. No new correctness, accuracy,
or required-gate defect was found in the changed implementation and evidence.

This review ran only source reads and offline artifact analyses. It did not run
tests, a server, clients, builds, device opens, hardware checks, or resets. The
only file written is this report.

## Required Work

None.

**Round 1 item 1: closed.** The original required 32-user reasoning-profile
gate now passes with its original temperature 1, top-p 0.95, top-k 40, seed 42,
128 output tokens per request, and ratio limit 1.15. Independently recomputing
the current raw result gives:

```text
78.23860912957764 / 82.33306772904251 = 0.9502695731812192
```

This replaces the previous 4.275190600 failure without a waiver or changed
profile. Both plain and profile runs complete all 32 requests with 128 tokens
each and no errors. The other 32-user ratios are 1.036004900, 0.946781826, and
0.949326704; every one-user ratio also passes. Evidence:
`doc/eval_readiness/harness_requests/runs.json`, the canonical
`harness_requests.json`, and `readiness_vllm/eval_probes/p2_repair/command.json`.

The declared capacity is backed by a real wider canonical sampler, not just a
routing declaration. The source keeps
`SamplingGenerator -> TTSampling -> ttnn.sampling`, selects 64 candidates per
shard, compacts the gathered values to global 64 on device, gathers matching
global IDs, seeds immediately before drawing, and retains the persistent
token-out binding. The TTNN opt-in requires width exactly 64, preserves both
sorted tiles, masks/normalizes both, and indexes both tiles in the writer.
The default 32 path retains its operation order and one-tile writer branch.
No adapter sampler or host token-feedback loop was added.

The raw P2 log prefix independently contains 55 device prefill and 2,562 device
decode forwards, with zero host forwards. Its byte range matches
`p2_final_async/p2_audit.json`. Thus the passing latency measurement exercised
the repaired device route. All 15 P2 checks pass; the current aggregate gate
reports eval-readiness pass and no critical failure.

## Other Concerns

- **Round 1 item 2: remains optional hand-off work.** Homogeneous `auto` versus
  `request` policy comparison, cohort overlap, batched prefill and chunk-budget
  tuning remain unmeasured and unclaimed. The wider sampler is now implemented
  and its required P2 result is closed. No additional performance experiment
  is required by this follow-up.
- **Round 1 item 3: remains closed by the prior appendix.** The current README,
  work log and acceptance notes distinguish the historical failed snapshot
  from the repaired result. Minor historical provenance remains in reused
  metadata; it does not contradict the actual current commands or results.
- **Round 1 item 4: measurement limits remain applicable and disclosed.** The
  primary performance workload contains one request, so its request-level
  percentiles are not tail-latency coverage. The mixed-shape lifecycle includes
  eager-prefill pauses and is separate from warmed performance. The retained
  max-context client's elapsed time is not server TTFT. Both new servers again
  used upstream zero-timeout abort and force-killed EngineCore after draining;
  cleanup records empty device-holder checks, but graceful trace teardown is
  still unproven.
- Current sampled qualitative outputs are an observational check, not a blind
  population-level comparison. The public-temperature correction intentionally
  changes stochastic behavior; retained seed-0 controls and unseeded serving
  outputs do not establish exact stochastic equivalence. The report makes no
  such claim. This limitation does not invalidate the exact greedy controls or
  the sampler's numerical/distribution evidence.

## Hard-Check Gaps

- **Round 1 item 5: retained, covered by independent evidence.** The current
  full sampling log reports 67 passed and 75 skipped in 543.59 seconds. The
  skips remain 71 speculative tests, three test-budget-dependent chunk tests,
  and one all-vocabulary logprob case. They are not passed coverage. The new
  same-server lifecycle supplies actual chunk continuation/tails, reordered
  rows, allocator-driven page growth and concurrent decode, so this does not
  leave a newly uncovered serving requirement.
- **Round 1 item 6: remains controlled.** The aggregate gate exits 1 for
  advisories: absent optional batched-prefill serial comparison and scans of
  historical prose containing the old `max_context: 0` metadata false positive.
  It reports supported context 262144, 84 valid serving-contract rows, no
  degeneracy, and passing P2/P3/P5. It is not evidence of a current context cap.
- **Round 1 item 7: remains a provenance limitation.** Scraped fields such as
  `chat_template: "that"` still occur in harness metadata. The pinned template,
  saved launch command and rendered prompt/token artifacts establish the
  configuration; those scraped strings are not used as configuration proof.
- The shared P2 device-line regex misses `operation=... mode=device`, so its
  per-profile device count is zero. The exact prefix log audit and this
  review's independent recount establish the actual route. Zero from that
  parser must not be interpreted as zero device work.
- `p2_final_async/sampling_qualitative_run_summary.json` contains merged stage
  metadata, including benchmark/prefill entries. Use
  `acceptance_commands.json`, the finished sampling/qualitative phase log and
  current output artifacts for that phase's provenance, and `p2_perf` for the
  measured benchmarks. The actual phase command exits 0 and runs the full
  sampling and qualitative stages; the metadata issue is not missing evidence.

## Anomaly Ledger

- Observed anomaly: top-k40 formerly required the slow host route and failed P2.
  Evidence: round 1 raw results preserved under `p2_repair/prior_artifacts/`;
  current raw P2 results and exact device-route log prefix.
  Affected path: canonical stochastic sampling and 32-user reasoning load.
  Control or comparison: unchanged plain/profile workload; legacy32 probe
  never draws ranks 33 through 40, whereas the wider probe reaches them.
  Likely subsystem: one-tile candidate retention, mask and writer traversal.
  Investigation performed: inspected the complete sampler diff, focused probes,
  full-serving routing and timings; recomputed all P2 ratios.
  Resolution: fixed.

- Observed anomaly: Qwen previously passed public temperature T to a low-level
  reciprocal-temperature parameter.
  Evidence: generator diff, `host_probe_retry.log`, `adapter_width64.json`.
  Affected path: non-unit-temperature stochastic sampling and parameter reload.
  Control or comparison: public T0.5/1/2/0.7 produces device values
  2/1/0.5/1.4296875; stored public values remain uninverted. Host reload checks
  and repeated seeded TP4 sequences pass.
  Likely subsystem: generator-to-canonical-sampler parameter convention.
  Investigation performed: inspected upload and signature caching, numerical
  probe cases and actual generated output.
  Resolution: fixed; no exact comparison to historical stochastic streams is
  claimed.

- Observed anomaly: lifecycle raw events include prior P2 events on the same
  server, and the shared P2 regex misses this adapter's device marker.
  Evidence: `p2_audit.json`, `lifecycle_audit.json`, `lifecycle_interval.json`
  and the raw server log.
  Affected path: evidence attribution, not request execution.
  Control or comparison: independently sliced saved byte intervals and
  recounted markers/events. P2 has 2,617 device/0 host forwards; lifecycle has
  775 device/0 host forwards, 27 chunks, 759 decode events and 41 page updates.
  Likely subsystem: event scope and log parsing.
  Investigation performed: offline byte-range/event analysis.
  Resolution: controlled; cumulative raw lifecycle events are not counted as
  interval-only events.

- Observed anomaly: repetitive raw-token stress continuations, story pronoun
  changes, and 256-token explanation truncation remain visible.
  Evidence: actual current lifecycle tokens and all 12 qualitative texts.
  Affected path: raw continuation stress and finite-budget chat samples.
  Control or comparison: independently compared all 16 greedy lifecycle token
  lists with the saved synchronous control and all six greedy qualitative
  texts with `tt_pc04_retry.json` after terminal-token removal; all are exact.
  Historical HF controls also document truncation. The current sampled texts
  remain coherent and on topic, including the requested French translation.
  Likely subsystem: prompt type, finite output budget and ordinary generation
  behavior; no new serving corruption is evidenced.
  Investigation performed: read every current qualitative text, checked the
  exact controls, and avoided a population-quality claim from unmatched samples.
  Resolution: controlled for this serving follow-up.

- Observed anomaly: a hook temporarily replaced the active sampling-log
  pathname; tracking-disabled preparation warns about active-trace allocations;
  shutdown force-kills EngineCore.
  Evidence: `p2_repair/log_recovery.json`, complete sampling log, strict tracked
  probe/server evidence, both new server logs and cleanup records.
  Affected path: evidence capture, trace validation and drained shutdown.
  Control or comparison: the exact 18,201-byte stdout was recovered from its
  held original inode; the log includes every test and final 67/75 result.
  Strict tracked acceptance includes program-cache allocations, while the
  focused unsafe-allocation positive control raises as intended. Performance
  has 282 device/0 host forwards. Both process pairs exit and recorded fuser
  results are empty; no reset is recorded.
  Likely subsystem: pre-commit stash, disabled-tracker diagnostic and unchanged
  upstream abort policy.
  Investigation performed: inspected recovery provenance, probe implementation
  and results, phase completion, raw shutdown logs and cleanup artifacts.
  Resolution: controlled; graceful teardown remains a disclosed limitation.

## Scope Inspected

- Goal: `/home/ttuser/qwen36-27b/multigoal_logs/09-09-vllm.prompt.txt`.
  Follow-up contract: `doc/vllm_integration/stage_review_round1.md`, including
  its documentation-closeout appendix. The owner explicitly resumed stage 9;
  the previous experiment cap is not treated as a current stop instruction.
- Skills: installed `stage-review`, `vllm-integration`, `tt-device-usage`,
  `tt-enable-tracing`, `qualitative-check` and relevant model-bringup startup
  rules in package `tt-model-bringup/0.1.24`. No new plugin authorization is
  needed or assumed; PC01/04/05 were already approved.
- Source: the diff from `b527d5a` to `450c36d7` in model generator/adapter,
  `models/common/sampling`, and TTNN sampling API, device operation, factory,
  compute and writer; new focused tests and acceptance/cleanup helpers.
  The declaration, program attribute, compile-time capacity, buffer counts,
  masks, tile-face offsets, global-ID compaction, parameter upload and seed
  ordering were inspected together. No new allocation exemption or capture
  lifecycle was introduced.
- Focused evidence: `p2_repair/AUTODEBUG.md`, `AUTOFIX.md`, host/legacy/wide/TP4
  probe JSONs and logs, build/hook results and log recovery. The width64 result
  contains 96 k/p/T histograms and 48 legacy32 control histograms, plus 24
  traced parameter configurations. All recorded histogram errors satisfy their
  statistical bounds. At p1/T1, rank64 is actually drawn 17 times in the k64
  histogram. Strict replay checks cover both capacities, changed seeds,
  row remapping, sequential/arbitrary IDs and persistent output. The reduced
  TP4 adapter adds actual generator sampling/penalty traces and external cache.
- Served evidence: `p2_final_async` commands, server log, current P2, exact
  lifecycle interval, token controls, logits, top-k128 companion route,
  full sampling, actual qualitative texts/metadata and TT/HF controls;
  current root gate/benchmarks and `p2_perf` audit/log/commands/cleanup.
  Both raw runtime model precision configs equal the selected full JSON.
  Registration resolves to this adapter with all 64 layers and the external
  8224-page cache.
- Reuse boundary: unchanged greedy P3/P5 and the accepted once-only
  262111+33=262144 context request are retained. Model/cache layout/positions
  and selected precision are unchanged; the current full cache pool and
  prepared wider sampler fit, current greedy controls match exactly, and the
  new lifecycle exercises the affected sampler/cache interaction. These facts
  support reuse without an additional max-context run. Unchanged whole-stage
  work was not re-reviewed.
- Commands run: read-only `cat`, `sed`, `rg`, `tail`, Git diff/status/revision
  reads, and standard-library Python JSON/log analyses. Analyses recalculated
  P2 ratios, request/token completion, route counts, lifecycle events, exact
  control comparisons and benchmark arithmetic. No target test was executed.
- Repository state: tt-metal HEAD is
  `450c36d7c4864f69f2e8365b7c62e4ff3506e495`; existing untracked raw evidence is
  left untouched. The plugin is clean at
  `da1587235b84b13f2826cfd7e8b448b340322a54`; no plugin change occurs in this
  diff. Upstream vLLM is recorded unchanged. The source/evidence checkpoint is
  local; the stage owner must locally commit this review closeout. Per run
  plan, any later push belongs to the supervisor. This review performed no push.

Independently supported current results:

| Check | Result |
| --- | --- |
| P2 reasoning 32 users, top-k40 | Ratio 0.950269573, limit 1.15; all 32 x 128 tokens complete |
| Same-server lifecycle | 17 requests; 16 exact greedy controls; 27 chunks at nine actual start positions; 759 decodes; 41 page updates |
| Trace reuse | Five prepared IDs 0–4, one constant trace map, capture count 5 throughout the workload, no interval capture/retirement; program count 24888 remains constant |
| Late unsupported companion | Current companion k128; sampled and greedy active streams match solo exactly, and companion streams match their controls |
| Primary 128/128/1, concurrency 1 | TTFT 275.746611 ms; mean TPOT 26.572380 ms; derived 37.633061 t/s/u; output 35.060375 tok/s |
| Secondary CI 100/100/32 | 32 complete; output 229.093610 tok/s; derived 9.925721 t/s/u |
| Serial prefill 128/32/32, concurrency 32 | 32 complete; output 107.532538 tok/s; derived 5.914450 t/s/u |

All three normalized benchmark records agree with their raw results, including
percentiles, completion counts, zero missing output tokens and TPOT-derived
rates. They are refreshed measurements of the repaired snapshot, not reuse of
the earlier benchmark numbers.

## Residual Risk

- This is a focused follow-up review of finished evidence, not a soak or release
  certification. The wider sampler's numerical probe is on the target
  Blackhole hardware; no new cross-architecture certification is implied.
- The precise top-k/nucleus boundary and seeded replay probes substantiate the
  new path, but the short served quality suite is not a population-quality
  evaluation. Historical sampled controls do not validate the corrected
  temperature stream byte for byte.
- Optional policy/prefill tuning and graceful teardown remain hand-off topics.
  None is a new failed stage-9 requirement. No further experiment is required
  to close round 1 item 1 or complete this round-2 review.
