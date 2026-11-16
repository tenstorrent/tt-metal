# SWE85 final collection audit

Scope: host-only inspection of final collection, adjudication, learned task measurements and all 85 coordinator records, 2026-11-14. This is a scoped audit, **not a whole-stage clean-pass**. No inference, device access, tests, containers, score or roster mutations were performed. Precision investigation remains in progress.

## Findings to reconcile

- **P2: Requests1724 rerun status contradicts its recovery evidence.** Its coordinator and the corresponding `results.json` failure row retain `tried=false`, `required_pending_resource_slot`, and a request to wait for active workers. The same coordinator records attempts1/2; attempt2 executed original tests (86 pass, external HTTP429/TLS failures remain). `rerun_queue.json` already says `rerun_infrastructure_persists`. Synchronize the status and link the completed attempts; keep `counted=false` and the unknown outcome. This is bookkeeping repair, not a reason to run another experiment or award a score.
- **P2: learned extended-phase status is stale.** `learned.json.current_v5_phase_status.extended.status` still says `gpqa198_corrected_passed_swe85_running`, although all85 first attempts have finished and final collection exists. Update it to collected/below-band/analysis-in-progress with the final evidence link. Its85 per-task timing rows are current and need no replacement.
- **Schema consistency note:** nine accepted-pass coordinators omit `anomaly` (Django11299/11790/13363/14999, Astropy14309/14995, Matplotlib23412/24149, Flask5014). Do not interpret an absent flag as independently audited false. If normalizing the schema, set explicit flags from their existing review evidence. No positive digest flag was found paired with a false/missing anomaly flag. Many older coordinators omit `rerun`, but all62 current `results.json` failure rows have nonempty rerun records; this audit does not demand redundant coordinator fields where the authoritative failure row already records the decision.

## Re-derived score and accounting

All85 task sets agree across final score rows, coordinators, collection and call records. Every final row’s native reward, counted reward and primary cause matches its coordinator. The category totals are23 accepted passes,22 fetched solutions,26 wrong patches,11 primary anomalies and3 infrastructure-unknown outcomes. There are21 explicit anomaly flags because some are secondary to another primary cause; that does not contradict the11 primary-anomaly count.

Native31 successes include eight fetched-solution successes and are not the accepted score. The three unknowns (Requests1724/1766/2317) remain null, remain in the85 denominator, and are neither silently failed nor excluded. Recomputed point-score bounds are23/85 through26/85 (27.0588–30.5882%). Wilson lower bound at23 is18.7596%; upper bound at26 is41.0458%. This envelope is explicitly not an observed point-score confidence interval. Even its optimistic upper end is below the64.8% reference.

The collection contains4427 unique mapped response IDs:4397 native-response-ID assignments and30 exact-statement/unique-trial-interval recoveries absent from native trajectories. Totals independently sum to87066659 prompt tokens,83913152 cached,3153507 uncached and1202253 output tokens. All4427 requests retain the32768 output limit. All six length-capped responses are retained, each32768 tokens: Requests2931 call20, Pytest7432 call19, Sympy14248 call53 and Requests5414 calls9/10/11. The latter three alone account for98304 discarded-from-native output tokens, which remain in server accounting. No cap-driven score denominator deletion was found.

All85 per-task outside-model intervals reproduce as trial wall minus the union of mapped request intervals. The task-lifecycle wall range is3.5225030569h; adding3.421446085s global overhead once gives3.5234534586h. Independently, the exit marker contains0 and has mtime2026-11-14T21:34:07.432451Z; subtracting run start18:02:43Z gives the same3.5234534586h. The later supervisor end21:34:09Z/rounded3.524h is preserved separately and was not substituted.

The8.7897504017h replay is labeled approximate, with extrapolation beyond a batch32 measured context regime recorded. Measured wall is below the one-sided1.25× timing threshold; faster execution does not imply accurate rate interpolation. Do not describe the overall longest131072 row as eliminating every batch/concurrency-specific extrapolation. Separate verifier-recovery work is retained in coordinator evidence, not double-added to overlapping list wall.

All85 current-run `learned.task_times.swe_bench_verified` records match final call counts and prompt/cache/uncached/output sums. Their solved fields are23 true,59 false and3 null, consistent with the adjudication. Historical best-config entries remain historical and were not substituted as this run’s results.

## Triage reconciliation inspected

Coordinator decisions, rather than heterogeneous initial triager call indices, bind final controls. For Sympy14248, raw46/native92 is the unsuccessful sed attempt: native99 still displays the original printer body. Raw50/native100 writes the replacement; native103 displays the changed body. Raw53’s32768-token assistant response is absent from native messages, not an ordinary tool traceback. The coordinator’s correction of initial triage is therefore supported by raw source observations and response-ID mapping. Raw79 maps to native157. Requests5414 retains its three assistant caret collapses separately from terminal URL-regression evidence and secondary verifier environment errors. These distinctions must remain in the final control roster and anomaly report; they do not resolve precision causation.

## Evidence scope and remaining work

Inspected `learned.json`, `results.json`, `runs.json`, the actual exit marker, `current/swe-extended85-head-bfp8/collected-final/{final-score-with-adjudication,collected,calls,explained,digest_accounting}.json`, all85 `live-tasks/*/coordinator.json`, rerun queue, selected native trajectories/digests and recovery links. Raw task text is omitted here.

This audit validates arithmetic, retention and timing provenance at the collection boundary. It does not independently re-grade85 patches, resolve the three external verifier outcomes, endorse an unexplored precision explanation, or review final85 diagnostic fixture additions. Those and final stage review remain pending. The two stale status findings can be fixed without changing any measured result.

## Resolution addendum

Both P2 bookkeeping findings are resolved. Requests1724 coordinator and results now record `rerun.tried=true`, persistent infrastructure status, and both recovery provenance paths. The learned extended status now explicitly says SWE85 collected below band with controls in progress. These corrections change no scores or measured timing.

Independently recounted the retained server JSON sampling-mode events after probe completion line277 through last task completion line174256:426 host-compatible prefill,4288 device prefill and137686 device decode steps, matching `sampling_counts.json`. These are steps, not token counts or proof that every host-compatible batch mixes final/intermediate rows. Whole-stage closure remains pending.
