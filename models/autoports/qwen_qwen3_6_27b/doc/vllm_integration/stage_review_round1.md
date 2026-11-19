# Stage Review

Verdict: more-work-needed

Independent round 1 review of stage 9, Qwen/Qwen3.6-27B vLLM integration,
on 2026-11-16. One required gate remains failed. No additional blocking
correctness defect was found in the inspected serving path.

The latest owner instruction caps further experiments and permits carrying
non-blocking work. It does not waive P2 or permit a failed required gate to be
reported as passed. Preserve this implementation and evidence as a capped local
checkpoint; do not mark stage 9 passed. No test, server, client, hardware,
reset, or gate rerun was performed during this review.

## Required Work

1. **P1: P2's required 32-user reasoning-profile latency gate fails and is unwaived.**

   Evidence: `doc/eval_readiness/harness_requests.json` reports
   `ratio_lm_eval_reasoning_32u` as critical/fail. Recalculation from
   `doc/eval_readiness/harness_requests/runs.json` gives
   `356.79754386215996 / 83.45769281777751 = 4.2751906003583855`, against
   the required maximum 1.15. Both workloads complete 32 requests with 128
   output tokens each and no request errors. The reasoning profile requests
   temperature 1, top-p 0.95, top-k 40, seed 42. The real device candidate
   ceiling is 32, so it uses the explicit host route. The probe records 134
   host-sampling lines for that load run.

   `readiness_vllm/eval_probes/final_pass/status.tsv` records P2 exit 2 and
   aggregate eval-readiness exit 2. `readiness_vllm/stage9_gate.log` also retains
   this critical result. `doc/eval_readiness/limitations.json` explicitly
   records `fix_hours: 0`, `original_model: "not_tested"`,
   `waiver_requested: false`, and `status: "unresolved_after_owner_cap"`.
   The eligibility field `waivable: true` in the probe result is not a waiver.

   Why this matters: the original stage goal explicitly requires P2 to pass;
   the vLLM skill requires harness profiles to meet the load gate. A fast
   device-sampled headline benchmark does not establish this different
   harness workload's performance. The cap cannot turn this required failure
   into an optional wider-sampler optimization.

   Required next step: keep this gate open in the capped hand-off. If work is
   resumed, resolve the top-k40 route's cost in the canonical model/sampler
   path and rerun the affected P2 checks, or obtain an explicit, valid
   row-specific waiver/contract decision with supporting evidence. Do not
   silently reduce top-k, raise the declared ceiling without a working wider
   sampler, invent two hours of repair effort, or claim HF equivalence. No
   further experiment is requested during the current capped turn. This is
   not an outstanding plugin-interface authorization request: PC01/04/05
   are already approved and implemented.

## Other Concerns

2. **Optional performance work remains correctly unclaimed.** The work log's
   `Carried to next stage` section lists homogeneous `auto` versus `request`
   overhead, conservative lane overlap with paused host owners, batched
   prefill, and chunk-budget tuning. No same-harness policy comparison or
   batched speedup is established. These are hand-off items, not separate
   blockers for this capped integration review. Wider sampling is also a
   candidate implementation idea, but the failed P2 result it would address
   remains Required Work item 1.

3. **Some historical status prose remains stale.** PC05's entry in
   `plugin_changes.md` still says full-model acceptance is pending, and some
   serving-contract rows retain earlier pending/future wording even though
   current evidence is linked elsewhere. Final acceptance ran and failed
   only the recorded critical P2 check. Reconcile these status sentences with
   this review during documentation closeout; no serving rerun is needed.

4. **Measurement limits are disclosed and should stay explicit.** The primary
   benchmark has one request, so request-level P50 and P99 are identical;
   it is not a tail-latency characterization. The capacity client's 535-second
   timing is not server TTFT evidence. The 17-request mixed-shape lifecycle
   has long eager-prefill pauses and is a correctness/trace workload, separate
   from warmed serving performance. Graceful trace teardown was not proven:
   upstream's zero-timeout abort force-killed both EngineCore processes after
   the workloads drained, while the saved cleanup checks show devices freed.

## Hard-Check Gaps

5. **The shared sampling suite's skipped cases are not passed coverage.** Raw
   `sampling_tests.log` reports 67 passed and 75 skipped. Final gate notes
   account for 71 speculative tests, three chunked-prefill tests whose
   test-only budget flag was absent, and one all-vocabulary-logprobs test.
   The live lifecycle supplies actual changed chunk starts, partial tails,
   concurrent decode, and synchronous token controls; P5 supplies long-context
   recall. These independent artifacts cover the required serving behavior,
   so the three skips are not an additional blocker. Do not report the
   skipped plugin tests as executed.

6. **The aggregate context diagnostic was a metadata false positive.**
   `stage9_gate.log` interpreted a process return code named `max_context: 0`
   as a context cap. The summary now uses phase/exit-code records;
   `context_metadata_check.log` confirms supported context 262144. The actual
   launch command, `/models` response, and capacity result all agree. This
   corrected metadata finding does not remove the independent P2 failure.

7. **P3's scraped settings are imperfect provenance.** In raw
   `chat_conformance/responses.json`, fields such as `chat_template: "that"`
   are log-parser artifacts. Use the saved server command, prompt metadata,
   and rendered-token comparisons for configuration provenance. The actual
   responses and P3 checks do not show a template failure.

## Anomaly Ledger

- Observed anomaly: top-k40 reasoning load is 4.275 times the plain baseline.
  Evidence: P2 raw runs and failed critical check cited in item 1.
  Affected path: 32-user `lm_eval_reasoning` host compatibility.
  Control or comparison: same-server plain greedy load, same output count.
  Likely subsystem: real 32-candidate device sampling boundary and host path.
  Investigation performed: route/source inspection and independent ratio
  calculation; no invented repair-time waiver.
  Resolution: more-work-needed, Required Work item 1.

- Observed anomaly: historically, a late host-only companion changed a seeded
  device request's sampler and output.
  Evidence: PC05 reproducer records and final
  `pc05_final_async/topk_route.json`, plugin regression logs.
  Affected path: mixed host/device request admission and async lifetime.
  Control or comparison: final solo/mixed active and companion streams match
  exactly for both sampled and greedy checks; reduced cases broaden admission
  features, and host regressions cover scheduler/lane/replay transitions.
  Likely subsystem: plugin per-step routing versus request ownership.
  Investigation performed: inspected the immutable classifier, identity-bound
  owner storage, scheduler cohort selection before allocation, runner owner
  validation, and lane reload/drain changes.
  Resolution: fixed for the evidenced request policy; mixed-cohort performance
  remains a hand-off concern.

- Observed anomaly: raw-token lifecycle outputs contain repetitive numerical
  continuations; the chat story has a pronoun change and long explanations
  end at the 256-token budget.
  Evidence: lifecycle raw request text, all 12 final qualitative texts, TT/HF
  control artifacts.
  Affected path: raw-token stress and finite-budget qualitative samples.
  Control or comparison: all 17 lifecycle token lists independently match the
  synchronous controls; all six greedy chat texts independently match the
  same-code TT control after terminal-token removal. HF controls also truncate
  the long explanations. No mechanical collapse, unwanted language drift,
  control-token leakage, or request contamination appears in the chat suite.
  Likely subsystem: prompt type, finite output budget, and ordinary generation
  variation, not a demonstrated serving regression.
  Investigation performed: read the generated text and recomputed the exact
  comparisons. No population-level quality inference from unmatched sampled
  seeds or fewer than ten blind matched samples is made.
  Resolution: controlled for serving regression; general model-quality limits
  remain outside this small observational sample.

- Observed anomaly: tracking-disabled preparation warns about active-trace
  allocations; shutdown force-kills EngineCore.
  Evidence: both final server logs, tracked command, positive-control log,
  lifecycle events, benchmark audit and cleanup JSONs.
  Affected path: trace preparation and process teardown.
  Control or comparison: same full64 policy/cache/signatures pass strict tracked
  acceptance including program-cache allocations; representative positive
  control raises on a live unsafe allocation. Performance has 282 device and
  zero host forwards in its audited interval. Both processes exit, saved fuser
  results are empty, and no reset is recorded.
  Likely subsystem: generic warning with tracking disabled; unchanged upstream
  abort policy at shutdown.
  Investigation performed: inspected trace preparation/replay and absence of
  broad allocation exemptions; separated shutdown from request-time events.
  Resolution: controlled for this stage; graceful teardown is unverified.

## Scope Inspected

- Goal: `/home/ttuser/qwen36-27b/multigoal_logs/09-09-vllm.prompt.txt`, including
  the explicit final owner cap supplied to this reviewer.
- Skills: installed `stage-review`, `vllm-integration` and its request/prefill/
  plugin-change references, `qualitative-check`, `tt-enable-tracing`, and the
  applicable device-use rules under
  `/home/ttuser/qwen36-27b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/skills/`.
- Snapshot: tt-metal branch `main`, HEAD
  `b395aed696938618da4db986be1c59c5d6b42550`, frozen implementation plus stage
  evidence/documentation awaiting local checkpoint. Plugin branch `main`,
  clean HEAD `da1587235b84b13f2826cfd7e8b448b340322a54`, base
  `042a94863ad27d88207bcf71abdf67122c6f4c27`. Installed upstream
  `vLLM 0.31.0+empty` matches the pinned plugin installation script; upstream
  is recorded unchanged. Per run plan, stage commits are local only and the
  supervisor owns any later push; no push was performed by this reviewer.
  At review closeout the owner reported the local source/evidence checkpoint
  `98aa9992e837570df6ed57ae9455a32a0dee9f03`, with hooks passing and no
  implementation change during this review. This report follows that checkpoint.
- Implementation: `tt/generator_vllm.py`; generator cache validation,
  preparation, split-sampling, capture, page binding and decode methods;
  current plugin changes in config/platform, model runner/input batch,
  scheduler/lane scheduler, `request_sampling_policy.py` and
  `sampling_cohorts.py`; plugin interface/authorization records and host logs.
- Evidence: stage README/work log, acceptance summary/manifest/final notes,
  AutoFix/source-audit records, serving/context/precision contracts;
  `pc05_final_async` raw lifecycle and server log, audit, controls, logits,
  route, context and commands; representative adapter/penalty tracking
  artifacts; final qualitative texts/metadata and TT/HF controls; final
  sampling log; P2/P3/P5 raw and normalized results; all three benchmark raw
  and normalized JSONs; `pc05_perf` log/audit/command; cleanup and gate logs.
- Commands: read-only `cat`, `sed`, `rg`, `tail`, `git diff/status/rev-parse`
  and small Python artifact analyses. Analyses re-extracted lifecycle and
  sampling events by saved byte range, compared token lists and greedy text,
  recomputed P2 and benchmark arithmetic, inspected runtime precision and
  registration, and read P3/P5 responses. The only file written is this report.

Key independently supported results:

| Contract | Evidence and conclusion |
| --- | --- |
| Canonical device path | Adapter delegates to `QwenGenerator`; `_sample_graph` binds `tt_out_tok=self.tokens`; `_execute` is nonblocking; async read owns separate host storage. Host full-logits output requires the explicit compatibility route. |
| Cache and precision | Exact external-cache identity checked; physical pool 8224 pages; fixed recurrent state retained. Both final server precision configs equal `selected_precision_config.json`, with 64 runtime layer/state entries, BF8 KV and BF16 logits. |
| Registration and authorization | Both EngineCore logs resolve this autoport adapter with layers 0 through 63. PC01/04/05 match recorded approvals; PC02 slot repair and PC03 import-order repair add no shared interface. Host suite: 2784 passed/16 skipped; explicit plugin hooks pass. |
| Trace reuse and async state | Raw lifecycle interval: 27 chunks at starts 0/128/256/512/640/768/1024/1152/1536, 759 decode events, 41 page updates, 775 device/0 host forwards; program-cache count remains 24885. Five startup capture signatures remain, with no interval capture or retirement. All 17 requests match synchronous token controls; consumed page growth, unchanged tables, row reorder and mode transitions are evidenced. |
| Context and prompt tails | 262111 prompt + 33 generated = 262144 total; prompt modulo32 is31; max-model-len is262144. No context reduction. Additional nonaligned lifecycle prompts include129/130/131/1027/1539. |
| Quality and API | Full sampling67pass/75skip; all12 chat outputs read; six exact greedy TT controls. P3 has15 passing checks, with actual correct content/thinking separation, tool argument types, history, stop and JSON responses. All nine P5 needle answers equal their target at nominal8K/32K/128K and depths0.1/0.5/0.9. |
| Prefix and batched prefill | Prefix caching has the allowed `recurrent_state` exclusion. Final serving uses serial prefill. The serial burst artifact exists; no batched equivalence or off/on speedup is claimed or needed for an unadopted optional path. |
| Primary performance | Raw128/128/1/concurrency1: TTFT282.487ms, meanTPOT26.491981ms, derived37.747271t/s/u, output throughput35.093858tok/s. Normalized values match raw. |
| Secondary performance | CI100/100/32 completes32: output226.867105tok/s, derived9.820144t/s/u. Serial prefill128/32/32/concurrency32 completes32: output106.577782tok/s, derived5.887401t/s/u. Raw/normalized metrics match; neither is the headline decode workload. |
| Cleanup | Both final server pairs exited; saved device-holder checks empty; no reset. Force-kill at drained upstream shutdown is disclosed. |

## Residual Risk

- This is inspection of finished evidence, not a rerun or full product release
  certification. Lane-DP and rank-local generalized policy paths have host
  regression evidence; the hardware target is one TP4 replica.
- The one-request primary number and short lifecycle are limited coverage,
  with no new soak, profiler, policy sweep or batched-prefill adoption implied.
- P2 remains the sole blocking finding found in this review. A later local
  checkpoint commit can preserve the work, but cannot itself close that gate.
  Passing P3/P5, sampling and performance artifact existence does not justify
  `clean-pass` or `pass-with-carried` while item 1 remains unresolved.

## Documentation Closeout Disposition

Item 3: closed for the reviewed status corrections. Inspected only the
documentation diffs in `plugin_changes.md`, `doc/full_model/serving_contract.json`,
the stage README and work log. PC05 now distinguishes completed acceptance
execution from the failed P2 gate; cache, penalties and async rows cite the
finished full-model evidence; README/work-log disposition preserves the sole
P2 blocker and this `more-work-needed` verdict. These edits agree with the
evidence already reviewed. No source, test, hardware, or full stage re-review
was performed. Required Work item 1 remains open; verdict unchanged.
