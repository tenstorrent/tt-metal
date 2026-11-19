# Final acceptance reporting notes

Current owner-resumed snapshot after canonical width64 repair. P2's round1
blocker is fixed; no waiver or relaxed workload was used. Current evidence is
in final_acceptance_summary.json and p2_artifacts.json. Historical PC05 notes
are preserved in p2_repair/prior_artifacts/doc/vllm_integration/.

- P2 all15checks pass;32-user top-k40/T1/p.95/seed42 ratio0.950269573<1.15.
  Raw doc/eval_readiness/harness_requests/runs.json and current summary
  readiness_vllm/eval_probes/p2_repair/harness_requests.json. Actual prefix
  audit2617device/0host; shared P2 device regex misses this adapter marker.
- Current full sampling67passed/75skipped in543.59s. Skips:71 speculative,
 3 chunk tests without their test-only budget flag,1 all-vocabulary logprob.
  Current lifecycle independently covers changed chunks and concurrent decode.
- Current lifecycle17requests,16 exact synchronous greedy controls,27chunks,
 759decodes,41page updates,2 pending-prefill concurrent-decode witnesses,
  five stable trace IDs0–4,0 workload captures/retirements,24888 stable programs.
 31 consumed new-page examples,26 page-only updates,718 unchanged-table steps.
  Byte-scoped evidence: p2_final_async/lifecycle_interval.json. The larger
  lifecycle.json also contains prior P2 server events; do not count those as
  lifecycle events. All775 lifecycle sampling forwards are on device.
- Current fixed logits and top-k128 late-companion ownership checks pass.
  Top-k64 is now supported, so the old64 unsupported-companion case is historical.
- Read all12 current qualitative texts: coherent/on-topic; no mechanical
  repetition, gibberish, unwanted language drift or contamination. Six exact
  greedy TT controls. Pronoun shift and256-token truncation disclosed; no
  complete-answer, population quality or seeded unseeded-stream comparison.
- P3 all15checks and P5 all9needle cases remain from final_pass. Both are greedy;
  model/cache/layout/positions/precision unchanged and current greedy controls
  exact. That directory's failed P2 is historical, superseded by p2_repair.
- Once-only max context262111+33=262144 accepted in pc05_final_async; unchanged
  full cache/layout/context fits current server with prepared sampling buffers.
  No repeat capacity request and no performance inference from its client time.
- Prefix caching has the allowed recurrent_state exclusion. Batched prefill is
  disabled; the current serial burst is a baseline, no off/on speedup claim.
- Final performance current p2_perf, allocation tracking0/telemetry0, decode
  tracing on. Primary128/128/1/concurrency1: TTFT275.746611ms, TPOT26.572380ms,
 37.633061t/s/u, output35.060375tok/s. CI100/100/32 uncapped: output229.093610tok/s,
  TPOT100.748347ms,9.925721derivedt/s/u; secondary only. Serial prefill128/32/32
  concurrency32: output107.532538tok/s, TPOT169.077425ms,5.914450derivedt/s/u.
  README and normalized JSONs record all percentiles and exact raw paths.
- Both current servers drained and exited; fuser empty, no reset. Upstream
  force-killed EngineCore at zero-timeout shutdown; graceful teardown unproven.
  Performance audit282device/0host, no OOV/errors. Optional compatibility host
  requests are restricted to diagnostic/shared tests, not headline measurement.
- Source build and hooks pass. During sampling, pre-commit replaced the tracked
  log pathname; exact original stdout was recovered from the still-open inode.
  log_recovery.json records provenance. No test-output reconstruction or rerun.

Aggregate gate finished exit1: no critical failures. Context262144,84 serving rows, degeneracy and P2/P3/P5 pass. Advisories concern missing optional serial-reference comparison and historical false-positive context prose. Independent round2 review returned clean-pass with no required work remaining; see stage_review_round2.md. Stage9 is complete.
