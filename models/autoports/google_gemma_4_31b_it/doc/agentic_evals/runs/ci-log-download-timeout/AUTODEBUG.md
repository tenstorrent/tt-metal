# CI log retrieval timeout: transient failure resolved in saved observations

Both affected downloads recovered. No source defect or specific transport root cause is established; the exact redirect-versus-blob cause remains unknown. No speculative fix or further discrimination is needed for this recovered incident.

## Direct observations

All times below are UTC on 2026-11-07. Evidence is local only.

| Run / job | Successful fetch start → finish | Seconds | Snapshot bytes | Latest event timestamp |
|---|---|---:|---:|---|
| 42323448662 / 128253219243 | 04:13:32.521969 → 04:13:34.438649 | 1.916680 | 61,058,717 | 04:11:14.8385103 |
| 42332785769 / 128282923304 | 04:13:38.813049 → 04:13:39.901232 | 1.088183 | 21,054,778 | 04:10:43.5313599 |

Both observation records report retrieval exit 0 and observer exit 0. Independent local checks of each saved `current.log` confirmed strict UTF-8 decoding, a final newline, byte count matching metadata, and SHA-256 matching its observer record. The saved observer-state offsets equal the complete file lengths. Hashes match the existing private `bringup/artifacts/ci-log-download-timeout/recovery-check.json`. These are complete successfully downloaded snapshots, not proof of complete running-job output, current freshness, or job completion. The newest events lag those fetch completions by approximately 140 and 176 seconds.

Subsequent saved fetches also succeeded: run 42323448662 at 04:14:35.006187–04:14:36.742706 (1.736519 s), and run 42332785769 at 04:14:40.124842–04:14:41.286416 (1.161574 s), with the same respective snapshot hashes and lengths. Thus recovery is not inferred merely from an old preserved file.

`bringup/artifacts/stage12-gpqa80-observer.log` directly records three exit-124 attempts for run 42323448662 starting at 04:06:02.386558, 04:08:32.394772 and 04:11:02.456395, each lasting approximately 90 seconds, followed by the successful fetch above. For run 42332785769, the saved successful observer records include an intervening slow success at 04:08:13.492905–04:09:07.243483 (53.750578 s), and recovery by 04:12:37.495232–04:12:38.529130 (1.033898 s). The reported timeout period therefore must not be described as uninterrupted failure of both runs. The second run's exact exit-124 sequence and the contemporaneous normal job/rate API successes remain requester-reported context rather than independently reconstructed facts here.

## Source explanation and attribution limits

In `bringup/artifacts/stage12-watch-eval-log.py:19–39`, one `timeout 90 gh api .../jobs/{id}/logs` subprocess covers the entire retrieval and writes stdout to `next.log`. The timeout is a whole-command bound, not a measured redirect or body-transfer deadline. Metadata contains start/end and exit code, without per-hop timing, HTTP status, or partial byte counts on failure. Captured stderr is not retained in the failure record.

Only exit 0 promotes `next.log` to `current.log` and invokes the observer (:40–79). Failure preserves the previous successful snapshot (:80–82); the loop then sleeps 60 seconds (:86). This explains the approximately 150-second spacing between timeout attempt starts and why a file remaining on disk alone would not establish recovery. The successful exit records and matching newly processed snapshots do establish recovery. The downstream observer's separate 60-second timeout (:74) cannot explain retrieval exit 124: that exit is recorded from the earlier retrieval subprocess.

The available metadata cannot distinguish waiting for the log redirect from waiting for, or stalling during, the redirected blob response. Successful ordinary API calls would not isolate those phases. The local `42323448662/probe.log` is a partial 14,305,460-byte file without a trailing newline, but no sequence-matched status/timing record was established for it; it cannot assign the observer failures to a particular hop. No credential, cache, proxy, rate-limit, runner, or model failure is proven.

## Prior cache evidence: a separate freshness limitation

The requested discriminator is recorded under `models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/ci-log-cache-followup/RESULT.json` and `plan.md` (the effort name is `ci-log-cache-discriminator`). Its standard and unique-query requests on 2026-11-06 both succeeded and returned identical newest-event timestamps, ETags, last-modified values and content lengths; event ages were approximately 434 and 435 seconds. The response Date changed. One sequential pair did not distinguish every cache from upstream archive publication delay, and did not establish a freshness bound.

The prior `runs/ci-live-log-observability/RESULT.md` likewise records unchanged archived snapshots despite Range/no-cache requests and distinguishes archive retrieval from a live-console channel. These observations do not establish the cause of the later transfer timeouts. Recovery of retrieval does not remove that separate freshness limitation. Neither cache busting nor increasing the timeout is justified as a fix by these artifacts.

## Review and scope

Claims were rechecked against the observer control flow, timestamped successful records, local snapshot framing/hash/length, and prior cache metadata. No headline code bug survived that check. The lack of per-hop telemetry limits attribution; it is not evidence that telemetry caused the timeout. No additional probe is proposed because both downloads have recovered.

Only this nested report was written. Root reports were preserved. No network calls, credentials, signed URLs, raw log content in this report, source edits, devices, subagents, score audit, or ongoing-stage evidence changes were used.
