# Evaluation priority and provisional QB2 performance criteria

User-directed provisional evaluation-practicality criteria; throughput >=50% strict target and TTFT <=2x strict target. Both per-user and aggregate throughput retained.

Performance sufficient to begin evaluation debugging. Keep prepared BS8/C8 R1 configuration initially; adapt concurrency only if actual evaluation latency/progress requires it. No performance optimization gate before R1.

| Input / output | C | TTFT ms | User decode t/s | Aggregate decode t/s | Relaxed result |
| --- | ---: | ---: | ---: | ---: | --- |
| 128 / 252 | 1 | 66.2 | 41.48 | 41.48 | pass |
| 1024 / 252 | 1 | 185.7 | 41.14 | 41.14 | pass |
| 4096 / 252 | 1 | 634.4 | 40.74 | 40.74 | pass |
| 16384 / 252 | 1 | 2634.4 | 39.91 | 39.91 | pass |
| 32768 / 252 | 1 | 5606.9 | 38.87 | 38.87 | pass |
| 65536 / 252 | 1 | 12565.5 | 36.94 | 36.94 | pass |
| 131072 / 252 | 1 | 33672.8 | 33.32 | 33.32 | pass |
| 261892 / 252 | 1 | 95943.1 | 28.29 | 28.29 | pass |
| 4096 / 252 | 8 | 4538.7 | 15.89 | 102.40 | fail |
| 32768 / 252 | 8 | 40191.1 | 11.70 | 36.20 | fail |
| 131072 / 252 | 8 | 171965.0 | 1.91 | 7.72 | fail |
| 4096 / 252 | 16 | 9423.7 | 8.88 | 109.86 | fail |
| 32768 / 252 | 16 | 66860.5 | 4.50 | 34.68 | fail |

Eight single-request points pass; five batch-8/16 points fail. The 262K point uses the authorized measured261892+252 workload:95.943s TTFT against120s relaxed limit and28.285t/s against17t/s relaxed minimum. It is not a new measurement or a claim that262144+252 fits.

All original strict misses and historical source scopes are preserved. These are base mixed-precision measurements; final promoted FP8 source performance is not reclassified by this analysis.

Priority: qualify the final-source APC-off/chunk-on FP8 server, run R1 control2 and then all198; prepare/run full Terminal2.1 and an explicitly labeled full SWE cohort when Docker/sandbox access is established. APC pairs, warm rows, fairness and full-model MTP optimization are deferred unless needed for evaluation practicality. Accuracy thresholds, full-cohort requirements, and ownership gates are unchanged.

Current blockers are recorded in access_preflight.json. Unknown PID0 entries persist on all four devices. Physical-host SSH authentication is denied, Docker/Kubernetes access is absent, and runner inventory is HTTP403. No new evaluation result is claimed.

SWE execution decision: use the full 500-instance Verified cohort for initial evaluation/debugging through the supported TTI path, with exact cached revision/IDs and runtime provenance bound before execution. Report a Verified score without applying the unmatched Pro threshold. The refined Pro reference remains an acceptance-comparison gap, not a prerequisite for running Verified. See evaluation_plan.json and r1_commands.json.
