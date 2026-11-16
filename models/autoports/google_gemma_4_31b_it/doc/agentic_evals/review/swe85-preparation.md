# Stage Review

Verdict: clean-pass

**Limited scope: stage 12 extended SWE85 preparation only. This is not launch admission, an evaluation result, or a final stage verdict.** Reviewed the live `v5-extended-run` checkout at `30d59bb033bd09472618ea32be3d9b36f46dfefe` while GPQA198 remained active. No device, server request, container launch/build, source edit, or HEAD mutation was performed.

## Required Work

None within this preparation scope. The explicitly pending launch checks below remain mandatory before evaluation starts; this verdict does not satisfy them.

## Other Concerns

- The projection is close to its margin boundary: 9.5900584178 h × 1.25 = 11.9875730222 h, only 44.7 seconds below the 12 h admission threshold. This is a ten-archetype extrapolation, not a confidence bound. Actual unseen-task tails can exceed it. The margin audit correctly supersedes the historical double-margin selection of 65 without changing that historical evidence.
- Image preparation ended at 175,771,037,696 free bytes against a 170,000,000,000-byte reserve. The recorded buffer above reserve is only 5.77 GB; the fresh Docker-host disk check is consequential.
- Historical partial migration/build records retain pending or validating rows. Completion is supplied by the later named records, not by treating those rows as successful. `all85-readiness.json` is the current identity map.
- The collection plan correctly identifies that its older fixed10 verifier-recovery driver is unsuitable for SWE85. Any actual infrastructure recovery needs a run-specific, evidence-preserving driver; no speculative recovery is needed before launch.

## Hard-Check Gaps

These checks are intentionally deferred and are not preparation defects while GPQA is running:

- Finish GPQA collection and required gate/triage review; the coordinator must explicitly admit SWE85. Preserve the fixed-phase pass evidence and unchanged precision/rate basis.
- Recheck host RAM, Docker-host free disk >=170 GB, all 85 image IDs and definition hashes, and unused job/run destinations.
- Verify the actual live server PID/start lifetime, implementation provenance, precision, pool, health, and zero running/waiting requests before the unscored probe.
- Record fresh per-run sizing tied to that server lifetime: measured pre-pool/free-memory provenance and current pool accounting, 131072-token planning working set, 12 agents, and smallest validated holding batch B32. Full served context remains 262144; the working set must not become a conversation cap. Any restart invalidates prior lifetime evidence.
- Run the prepared response-log probe and require the exact response ID to pass `reply_digest.py --check-log` before scoring. Record probe identity and actual evaluation start/end; preserve raw logs and exclude only proven unscored calls.
- Recheck remaining stage budget and the 43200-second outer timeout.

The prepared launch command itself does not enforce every admission prerequisite. It must remain under the coordinator’s explicit checklist until these are recorded. Final task accounting, first-attempt scoring, generated-output triage, solution-fetch audit, and timing explanation cannot be reviewed before the run exists.

## Anomaly Ledger

- Observed anomaly: historical 65-task selection applied margin against an already buffered ten-hour target.
  Evidence: `setup/swe_extended_admission_margin_audit.json`, `setup/swe85-candidate/selection-corrected-margin.json`, and all rows in `setup/extended_projection_trajectory_replay.json`.
  Affected path: extended-list admission arithmetic.
  Control or comparison: independently selected the maximum count among every recorded projection satisfying raw hours ×1.25 <=12; result 85. Count 86 gives 12.2794480222 h.
  Likely subsystem: planning arithmetic.
  Investigation performed: recomputed selection and verified projection-source SHA256.
  Resolution: fixed in candidate selection; historical evidence retained.

- Observed anomaly: linked-copy pilot changed ancestor permissions; migration records also include interrupted preparation.
  Evidence: `linked-pilot-two-copy-mode-mismatch.json`, `linked-pilot.json`, `Dockerfile.offline-linked`, `migration-pilot.json`, `migration-batch.json`, `migration-sklearn13135.json`, `migration-final4.json`, and the three `linked-*-preparation.json` records under `setup/swe85-candidate/`.
  Affected path: offline task-image preparation.
  Control or comparison: linked Dockerfile differs from original only by two COPY --link directives, a base-root-mode argument, and restoring that mode. Completed migration old/new Docker Config objects match. Recorded validation checks ancestor modes, original repository HEAD, runtime/package versions, and payload hashes; linked preparation verifies official base-layer prefixes and shared payload layers.
  Likely subsystem: Docker linked-layer directory creation.
  Investigation performed: inspected recipes/preparation code and completed migration records; matched all 85 live Docker image IDs to readiness via read-only `docker image inspect`. The incomplete sklearn row is completed by its separate successful record. New20 completion comprises 15+2+3 successful tasks.
  Resolution: controlled by corrected recipe and completed validation evidence.

## Scope Inspected

- Goal/skill paths: root `AGENTS.md`, `bringup/run_contract.json`, `bringup/agentic_evals/inputs.json`; installed model-bringup startup, stage-review, agentic-serving, and tt-device-usage skills at tt-model-bringup 0.1.27. User/operator local-only decisions override stale CI/credential statements in the run contract. Push state: local commits only; no push authorized.
- Artifact paths: `doc/agentic_evals/learned.json`, fixed-phase status in `results.json`; `setup/swe_extended85_launch_metadata.json`, `swe85_collection_plan.json`, `swe_extended_admission_margin_audit.json`, `extended_projection_trajectory_replay.json`; SWE65 frozen ordering and original definitions; SWE85 selection, combined manifest, all85 readiness, definition equivalence, migrations, linked preparation, and launch configuration.
- Code paths: `setup/swe85-candidate/Dockerfile.offline-linked`, `prepare_linked_next20.py`, original Dockerfile, all85 task definitions/test scripts, `setup/check_swe_extended85_response_logging.py`, and decoded collection recipes.
- Commands run: read-only file inspection and Python artifact analysis; `git rev-parse HEAD`; one batched read-only `docker image inspect` for all85 tags. An initial analysis invocation failed because the system Python lacks tomllib; rerun omitted that unused import. A migration analysis encountered an incomplete historical row and then explicitly followed its completion artifact.
- Independently verified: 85 unique task IDs; exact frozen500 prefix; launch-config task order; all recorded instruction/config/script/TOML hashes; config/combined-manifest/readiness/server-identity metadata hashes; all85 image IDs; largest feasible projected count across all recorded counts. For every task, original instruction and tests/config are byte-identical, task.toml differs solely in image selector, and test.sh has one common generic parser-construction correction. No task resources or repository test commands changed. The20 combined TOML differences from intermediate definitions are solely the recorded linked-image selector.
- Launch settings inspected: one attempt, 12 concurrent trials, unchanged main harness/mini-swe2.2.8 recipe, 21600-second agent and 3600-second request limits, 32768 output cap, temperature1/top_p0.95/top_k20, thinking enabled, local endpoint exception, and no resource override. The server default top_k32 operator decision does not override the harness’s explicit top_k20.

## Residual Risk

This review validates preparation and internal provenance, not future task success or total wall time. The projection itself retains disclosed limitations: synthetic active concurrency, unseen tasks, and unpriced exceptional setup/recovery. Fresh admission evidence must be collected after GPQA; source, precision, pool, image, or harness changes require checking affected preparation claims again. Evaluation outputs and final stage acceptance remain unreviewed.
