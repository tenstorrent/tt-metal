# Stage Review

Verdict: clean-pass

This verdict applies only to the user-directed QB2 performance regrading, evaluation priority, and reporting policy. It is not a performance acceptance pass, a completed evaluation, an authorization to bypass ownership checks, or a Stage 11 release pass. The original Stage 11 failure remains recorded.

## Required Work

- None in this bounded policy/documentation review. Runtime qualification, safe device ownership, sandbox access, and the requested full evaluations remain execution work; this review does not certify their completion.

## Recalculated Evidence

Independently parsed all 13 strict cold targets from the hash-matched Rev 0.11 CSV and compared them with `acceptance_matrix_before.json`. For every selected measurement, verified the pinned summary SHA256, raw-result SHA256, raw completion counts and lengths, metadata concurrency and sample count, historical source-inventory SHA256, and base-checkpoint precision scope. Recalculated mean TTFT and mean TPOT from `request_timings.json`, per-user decode as `1000 / mean_tpot_ms`, and aggregate decode as post-first-token output count divided by the union of the recorded client decode intervals. All checks passed.

The provisional tests are independently applied as `TTFT <= 2 * strict_max`, `user_decode >= 0.5 * strict_min`, and `aggregate_decode >= 0.5 * strict_min`, with successful execution and exact lengths retained. Values below are measured / provisional limit; TTFT uses an upper limit and both throughput columns use lower limits.

| Actual input / output | C | N | TTFT ms | User decode t/s | Aggregate decode t/s | Provisional | Historical strict |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 128 / 252 | 1 | 16 | 66.171 / 120 | 41.478 / 25 | 41.478 / 25 | pass | fail |
| 1024 / 252 | 1 | 16 | 185.746 / 300 | 41.142 / 25 | 41.142 / 25 | pass | fail |
| 4096 / 252 | 1 | 16 | 634.390 / 1000 | 40.738 / 24.5 | 40.738 / 24.5 | pass | fail |
| 16384 / 252 | 1 | 16 | 2634.369 / 3600 | 39.915 / 23.5 | 39.915 / 23.5 | pass | fail |
| 32768 / 252 | 1 | 16 | 5606.901 / 7000 | 38.874 / 23 | 38.874 / 23 | pass | fail |
| 65536 / 252 | 1 | 16 | 12565.478 / 16000 | 36.945 / 21.5 | 36.945 / 21.5 | pass | fail |
| 131072 / 252 | 1 | 16 | 33672.846 / 44000 | 33.323 / 20 | 33.323 / 20 | pass | fail |
| 261892 / 252 | 1 | 2 | 95943.052 / 120000 | 28.285 / 17 | 28.285 / 17 | pass | unverified nominal 262144+252 |
| 4096 / 252 | 8 | 32 | 4538.715 / 1000 | 15.890 / 17 | 102.405 / 136 | fail | fail |
| 32768 / 252 | 8 | 32 | 40191.052 / 7000 | 11.700 / 14 | 36.199 / 112 | fail | fail |
| 131072 / 252 | 8 | 32 | 171964.951 / 44000 | 1.914 / 11 | 7.718 / 88 | fail | fail |
| 4096 / 252 | 16 | 64 | 9423.719 / 1000 | 8.879 / 16 | 109.861 / 256 | fail | fail |
| 32768 / 252 | 16 | 64 | 66860.472 / 7000 | 4.496 / 12 | 34.681 / 192 | fail | fail |

Exactly eight rows pass and five fail. Each of the five failing rows fails all three provisional metric comparisons. The historical strict labels remain twelve failures, zero passes, and one unverified nominal 262144+252 row.

The authorized 262K substitution uses `../capacity/largest-valid-252/summary.json`, SHA256 `7c5961c662d5c47bd0ddc8a7aac6eae7275669def51c84baa2d03c7c96a0f138`. Its raw records contain two completed requests, each with 261892 input and 252 output tokens; one separate warmup is supported by server token-counter deltas. The total is 262144. This does not establish that 262144 input plus 252 output fits, and the original unresolved row has not been overwritten.

## Other Concerns

- The sufficiency decision is a provisional decision to begin evaluation debugging under the user's new priority, not a claim that the complete provisional performance matrix passes. Retaining initial BS8/C8 and changing concurrency only if observed evaluation progress requires it is consistent with that scope. Historical 252-output benchmarks do not establish long-output R1 throughput or completion time.
- All thirteen measurements identify the base `Qwen/Qwen3.8-27B` checkpoint with mixed BFP4/BFP8/BF16 and recorded server checkpoint `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. They do not requalify the current promoted FP8 server. The README and JSON preserve this distinction and require actual final-source/native/server qualification.
- `evaluation_plan.json` prioritizes diagnostic R1 control2, full R1 198, Terminal-Bench 2.1 full 89, then SWE-bench Verified full 500. R1 remains at 84.7% with at least 168 correct; Terminal remains at 69.4% with at least 62 correct. The supplied cohort metadata contains 89 distinct Terminal task IDs and 500 distinct Verified instance IDs; the cached Verified dataset hash matches its cohort manifest. These are preparation checks, not evaluated results.
- Running the full Verified cohort without waiting for a refined Pro reference is consistent with the debugging priority. Reporting must remain labeled Verified. The unmatched Pro 58.6% threshold remains inapplicable to Verified, and the original matched-Pro comparison remains unresolved. The new plan states this explicitly; no accuracy pass or waiver is implied.
- Full-cohort ID/outcome reconciliation, retained failures/timeouts, actual source/context qualification, and ownership gates remain mandatory. The R1 command manifests require the whole-client observer and native runtime binding; control2 remains diagnostic only. Their pinned spec and ID file hashes match. Detailed runtime qualification is outside this policy review.
- `access_preflight.json`, captured at 2026-09-16 08:52:41 UTC, records two unknown PID0 entries on each of four devices, missing local Docker/Kubernetes access, SSH authentication denial, and runner-inventory HTTP403. It supports the recorded launch/access blockers. This review did not retest live availability, and does not authorize running through those blockers.

## Hard-Check Gaps

- No missing evidence was found for the scoped arithmetic, historical labels, or documented evaluation priority.
- Final-source serving performance, full evaluation results, sustained isolation, and Terminal/SWE sandbox execution remain unverified. They must not be inferred from this review or from the frozen base-checkpoint summaries.
- This review checks existing client timing conventions; it does not claim that scheduler occupancy or client interval overlap measures physical active decode rows. The original acceptance snapshot preserves that limitation.

## Anomaly Ledger

- Observed anomaly: all five concurrent cold rows fail even the provisional bounds. Evidence: independent recalculation above. Affected path: historical base BS8/BS16 cold serving. Control or comparison: unchanged CSV strict targets and explicitly scaled provisional limits. Likely subsystem: serving performance. Investigation performed: raw timing, length, hash, and threshold checks. Resolution: controlled for this policy decision by the user's instruction that performance attainment is not an evaluation prerequisite; no performance pass is claimed.
- Observed anomaly: the nominal 262144+252 row exceeds the recorded total-context budget. Evidence: original matrix and capacity metadata/raw results. Affected path: 262K comparison. Control or comparison: authorized measured 261892+252 substitution. Likely subsystem: requirement interpretation. Investigation performed: exact request lengths, count, token accounting, and separate historical status verification. Resolution: controlled by explicit nominal/actual fields and retained unverified historical label.
- Observed anomaly: unknown owners and unavailable sandbox/host access prevent current execution. Evidence: access preflight. Affected path: new runtime/evaluation launches. Control or comparison: retained launch prerequisites and unchanged ownership gates. Likely subsystem: external execution environment. Investigation performed: inspection of captured preflight and documented launch policy only. Resolution: unresolved execution work, accurately retained; no waiver or evaluation result is claimed.

## Scope Inspected

- Contract: the user's latest instruction supplied in the review task makes strict performance targets stretch targets, sets provisional throughput to at least 50% and TTFT to at most twice target, authorizes measured 261892+252 for the 262K point, and prioritizes full R1/Terminal/SWE evaluation debugging without performance attainment as a prerequisite.
- Skill: `/home/mvasiljevic/qwen38-full-rerun/codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.4/skills/stage-review/SKILL.md`, plus its model-bringup startup instructions. Enabled plugin entries and packaged environment validation were inspected successfully.
- Target: `Qwen/Qwen3.8-27B` / promoted FP8; autoport `/home/mvasiljevic/qwen38-full-rerun/tt-metal/models/autoports/qwen_qwen3_8_27b`.
- Worktree: live, dirty checkout on `mvasiljevic/qwen38-full-bringup`, observed HEAD `1807cf6c7bf9616bb2545cd674d0469a94a3fde3`. Implementation files were not changed or certified by this review.
- Artifacts: all seven input documents below; all thirteen referenced measurement summaries and neighboring raw, timing, and metadata files; three pinned server source inventories; hash-matched original CSV; Terminal and Verified cohort metadata and cached Verified source hash; R1 pinned spec and ID hashes.
- Commands run: local `cat`, `sed`, `rg`, `git status --short`, `git branch --show-current`, `git rev-parse HEAD`; packaged `scripts/environment.py` with explicit installed plugin roots; small standard-library `python -` analyses for hashes, CSV parsing, metric reconstruction, comparison gates, and cohort uniqueness. One `rg` lookup for optional baseline notes used nonexistent paths; this did not affect the checks, which used actual measurement metadata instead. No network call, server, device operation, evaluation, or implementation test was run. Only this report was written.

| Reviewed input | SHA256 |
| --- | --- |
| README.md | `7bdeaeea46a3206c8834b8797b62fa23fd89a5b47586b7073a3272f9c2c2d062` |
| relaxed_performance.json | `375d1b07b4338e77e82d35fd144eee36e20cd41415167c786ef657cfd40d8b3a` |
| acceptance_matrix_before.json | `a9c29a7afded1af6698e9ec461678c920373cd98cb26348c180789fd33f1ec15` |
| access_preflight.json | `ac7af6733e80c9c00a1edf1aa40b89bed4fc2e1df1df53ed54fe0dabf7e0764e` |
| evaluation_plan.json | `1bfa0459d4f4ee379f9e52973ad63dc70c938758095205961096acf729e104d2` |
| r1_commands.json | `52f2450fe360829cedccc4c578441fc0a74ca4bcf23da63fc1f79f7a1b5c3de1` |
| r1_preparation_check.json | `9f0cd6c49c4fe7733ede7bdddcad8035f00221f4f498e3aae8479720cabc66cd` |

## Residual Risk

The current FP8 source may behave differently from historical base measurements, and actual long-output evaluation throughput remains unknown. Dataset manifests alone do not prove task execution or scoring. Device isolation and sandbox access still require resolution before affected launches. The remaining accuracy, feature, and original Stage 11 gates retain their existing status; this bounded clean-pass cannot close them.
