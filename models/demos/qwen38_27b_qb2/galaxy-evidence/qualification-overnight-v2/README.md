# Qualification and release readiness, Oct 9 2026 UTC

This is an experimental branch, not a qualified release. Full GPQA must reach
177/198 for the unchanged 89.2% gate. No passing full reference evaluation is
claimed for the optimized model. No Qwen release image or tested Helm deployment
has been produced by this work yet.

| Full GPQA | Correct | Truncated | Measured duration |
|---|---:|---:|---:|
| Historical native, 32K output | 171/198 (86.36%) | 5 | 27m58s |
| Shared Q/K candidate, 32K output | 142/198 (71.72%) | 42 | 32m13s |
| Shared Q/K candidate, 64K output | 163/198 (82.32%) | 15 | 57m35s |

All use the complete pinned Diamond set, concurrency 128, temperature 1,
top-p .95, top-k 20 and seed 42. The larger budget changes the experiment and
does not prove a numerical fix. The earlier native implementation and candidate
also differ in decode attention and other changes; their score delta alone
does not identify recurrence as the cause. The new native control retains
accurate attention and changes only the recurrence policy from the candidate.

The 64K run averaged 19.15 decode tok/s/user and 1,081 aggregate output tok/s
over the measured benchmark. These include a varied reasoning workload and
must not be substituted for steady-state, fixed-context kernel throughput.
Private full responses remain on the allocated host; public receipts contain
hashes, token usage and correctness, not GPQA examples.

## What stopped the first queue

`qwen38-overnight-v1-20261008.service` ran independently of the client session,
completed full GPQA, and stopped at 19:32:09 UTC after an HTTP sweep read error.
It did not execute native-control or extended-delivery stages. Its service is
terminal (`MainPID=0`, exit 1); this is not an observation timeout.

Tau3 attempted twelve processes, but all failed before inference because
`data/tau2/user_simulator/simulation_guidelines.md` was absent from the sparse
checkout. Its raw v1 summary reports zero successes, but **there is no valid
model score: zero model calls were made**. The corrected reporter emits null
accuracy for this case and retains the fixed denominator for measured trials.
The new preflight constructs actual upstream task metadata, including global
guidelines, rather than checking imports and task IDs alone.

The HTTP sweep completed 32K at 32/64/128 clients and 16K at 32/64 clients.
16K/128 failed; 128K and near256K remained unrun. Worker logs show the
supervisor's subsequent SIGTERM cleanup, not a proven earlier device crash.
Fresh connections are used for the next sweep to test the stale connection
hypothesis. No failed request is retried or removed from the old results.
If another HTTP failure occurs, it stays failed; progression to a fresh control
requires a healthy endpoint check and owned-worker shutdown. Hardware or
readiness failures still stop dependent stages.

## Resumed persistent queue

Host: `ttuser@10.228.203.98`. User service:
`qwen38-overnight-v2-20261009.service`, launched 03:28:38 UTC.
Source: `/home/ttuser/qwen38-artifacts-20261007/overnight-source-v2`.
Results: `/home/ttuser/qwen38-artifacts-20261007/overnight-v2`.

1. Native-recurrence G0 on all eight physical TP4 groups.
2. Full 198-question GPQA at 64K output, identical sampling and fixed precision.
3. Corrected official Tau3 pilot: twelve preselected banking tasks, eight
   concurrent tasks, one attempt, 60 steps, 20 minutes per task and 45 minutes
   total. Agent: 8K output, thinking, T1/p.95/k20. User and assertion verifier:
   local Qwen, greedy, no thinking, 2K output. This is not a matched published
   reference setup. Rewards, tool format errors, truncations and timeouts are
   reported separately; raw calls are retained before the upstream tool parser.
4. Real Galaxy HTTP sweep: 32K,16K,128K,262016 ISL; offered concurrency
   32/64/128 where the current KV pool permits it, one warmup and three bursts.
5. Fifty-four delivery variants: three receiver placements, packet sizes
   4/8/15 pages, ring depths 2/4/8, and consumer delay 0/4096 cycles.

The native control is fresh, not a rerun of already completed candidate GPQA.
CPU preflight passed **376 tests plus 40 subtests**. Upstream Tau3 source and
full metadata preflight passed. The service was observed live with PID 917206,
hardware reset completed, and native model layers loading at 03:30 UTC.
This snapshot is not a claim that the later evaluations passed.

The queue owns `/tmp/tt-device.lock` through safe runners, has a twelve-hour
hard limit, a 256-GiB host-memory limit and scoped process-group shutdown.
Linger is enabled. SSH/session disconnect does not stop it; automatic resume
after host reboot is not configured. Stop only the named user service to cancel.
All source, dependencies, logs and results are task-owned host-disk files.
Native runtime, checkpoint, firmware and NFS remain unchanged.

## Environment setup and preservation

Tau3 is pinned to `17e07b1da2bbc0cadfddeea36412686e0604127b`; the dataset hash
and fixed selection live in `tests/tau_benchmark.py`. Its isolated environment
uses the upstream frozen `uv.lock`, Python 3.12 and knowledge/voice extras
(upstream text imports audio support). PortAudio, ALSA and JACK dependency
packages were downloaded and extracted beneath `tau-environment-v1/portaudio`,
not installed in the host OS. Setup v1 omitted cone-mode root files; v2 exposed
a missing local ALSA runtime; v3 imported successfully. Shared simulator data
was then restored and the stronger metadata preflight passed.

Old results, source snapshots and failed setup receipts are preserved. The
launch manifest records exact commands and source hashes. Follow the actual
service and `queue.json`, not this point-in-time snapshot, for current progress.

The full profiler repair and new GDN epilogue/attention integration are not in
this queue: they need additional development. Passing diagnostic kernels does
not qualify those missing optimizations or the release.
