# Conditional GPQA-only no-score dry route

Conclusion: if the final go set contains only GPQA plus the separate local benchmark, the unchanged route exposes sufficient live evidence for a genuine early-stop dry run without completed scores or final sample artifacts. This is conditional, not a decision to stop SWE. It is also a future procedure, not a claim that the frozen release has already passed it. No device access, dispatch or polling performed (0 device hours).

GPQA is an HTTP lm-eval workload: it creates no Harbor task containers, images, Docker bridges or sandbox slots. Consequently the task-environment pull/build and simultaneous task-sandbox parts of the dry-run checklist have zero applicable objects for this suite. The model server still runs in Docker on CI and must prove its image/config and readiness normally. The selected number of simultaneous HTTP requests must be observed.

## What proves a reply

Pinned lm-eval source af401ef3dd3d2d90e55680990ed1f0c9f7892448, locally available at `/home/bringup/.cache/uv/git-v0/checkouts/29ebc9d5411336d3/af401ef3/`:

- `lm_eval/models/api_models.py:659-727`: amodel_call acquires the concurrency semaphore, POSTs, raises non-success status, awaits response.json for stream=false, parses generations, then returns answers. An exception is re-raised.
- `:820-874`: connector and semaphore use configured concurrency. `_track` awaits the retried request and only then increments `Requesting API`; exceptions do not increment it. Therefore a progress counter of k/N is stronger than access HTTP200: it proves k successful parsed request batches. Batch_size1 makes these k completed sample requests.
- `lm_eval/evaluator.py:580-608,637`: generate_until returns the whole request set before filters, process_results and score aggregation. A saved progress counter with0<k<N, while generation remains active, precedes the final suite score. No final sample artifact is needed to prove that at least these replies occurred.

An access200 alone proves status, not parsed content; with streaming it can even precede generation. Do not use it alone. Device active_rows proves concurrent generation, not delivered replies. The combination of configuredC, observed active_rows=C, and successful progress k>=C withk<N proves the configured concurrent workload ran and at leastC replies reached the client. It does not identify which initial request completed: faster slots can refill. If the claim specifically requires every one of the initialC request identities to have replied, aggregate counters cannot prove that without request-ID completion logs; do not make that stronger claim.

## Existing evidence of observability

`bringup/artifacts/stage12-gpqa-stop-snapshot.log:9550` preserves the full198 IDs, num_concurrent2, batch_size1, pinned revision, stream=false and unchanged limits. It shows full60 autoport registration at:1489 and active_rows2 at:9584 onward. Progress2/198 is preserved at12:34:24UTC (:16757), while10/198 appears at12:51:49 (:48495), well before all198 would finish. `runs/full-gpqa-observable/stop.json` records the later stock cancellation and retained live snapshot/run/job metadata; no final evaluation artifact was uploaded. This demonstrates the evidence channel survives stock cancellation when collected before it. It is not a dry proof for a different final tasklist/SHA.

`bringup/artifacts/stage12-gpqa-server-sized-live.log:9539` preserves the10 selected IDs and num_concurrent2. Progress2/10 is visible at11:02:14 (:20123); the final collected results timestamp is11:12:18. This gives an observed window to detect early replies before the score. Do not assume identical timing for a futureC4/frozen run.

## Concrete future stop checklist

1. Freeze chosen GPQA task IDs, C2orC4, serving settings, limits, same immutable tt-metal/TTI/plugin SHAs and route for dry and release. Record run/job metadata and the resolved image SHA. GPQA list must contain more thanC requests to permit early stopping before completion.
2. Dispatch the normal stock `workflow=evals` route with the frozen selected list. Preserve live console evidence of autoport class/path/full60, max_num_seqs and batch, pinned revisions, served defaults, endpoint readiness, complete lm-eval command including exact IDs, request/output limits, stream=false, batch_size1 and num_concurrent=C.
3. Preserve live adapter evidence active_rows=C after actual eval starts. Wait for the successful `Requesting API` counter k>=C and k<N, not merely its initial0/N line or HTTP200. Ensure no API failures/sentinel/error messages compromise the observation. Log lines can lag, so use the live source's actual timestamp and current generation evidence; if entire suite already finished, this attempt is not a no-score early-stop dry run.
4. In the separate authorized local benchmark arm on its frozen TTI/runtime/config, preserve actual successful request/response evidence that its benchmark client reaches the server. Use its supported launcher and exact matrix. HTTP-only benchmark introduces no task sandboxes either. Retain this arm's own clean dry record.
5. Save complete live log snapshot and metadata before stock CI cancellation; record observed condition/time, cancel command and final state. Do not depend on post-cancel artifact upload, score the dry attempt, or call its cancellation a release failure. Preserve any cleanup/recovery result from the stock workflow before reusing a runner through the pool.
6. Start release only on those same frozen settings/SHAs/tasklists with measured margin and queue projection. No completed-rehearsal exception is required if the actual early-stop conditions above are met.

Remaining uncertainties are operational: final go scope, frozen tasklist/concurrency, live collection latency, enough partial-progress window before completion, and actual future config/response evidence. A server-sized prior score alone cannot substitute. This result narrows the earlier general task-container dry-route limitation only because GPQA has no task-container environments to prebuild; it does not solve the analogous SWE/Terminal condition.
