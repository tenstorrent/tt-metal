# Qualification and release readiness, Oct 9 2026 UTC

This is an experimental branch, not a qualified release. Full GPQA must reach
177/198 for the unchanged 89.2% gate. No passing full reference evaluation is
claimed for either model. An experimental image is built and preserved on host
disk; container hardware qualification and a tested Helm deployment remain open.

| Full GPQA | Correct | Truncated | Measured duration |
|---|---:|---:|---:|
| Historical native, 32K output | 171/198 (86.36%) | 5 | 27m58s |
| Shared Q/K candidate, 32K output | 142/198 (71.72%) | 42 | 32m13s |
| Shared Q/K candidate, 64K output | 163/198 (82.32%) | 15 | 57m35s |
| Matched native control, 64K output | 170/198 (85.86%) | 1 | 48m39s |

All use the complete pinned Diamond set, concurrency 128, temperature 1,
top-p .95, top-k 20 and seed 42. The larger budget changes the experiment and
does not prove a numerical fix. The earlier native implementation and candidate
also differ in decode attention and other changes; their score delta alone
does not identify recurrence as the cause. The new native control retains
accurate attention and changes only the recurrence policy from the candidate.

The optimized 64K run averaged 19.15 decode tok/s/user and 1,081 aggregate output tok/s
over the measured benchmark. These include a varied reasoning workload and
must not be substituted for steady-state, fixed-context kernel throughput.
Private full responses remain on the allocated host; public receipts contain
hashes, token usage and correctness, not GPQA examples.

The [saved-response audit](../gpqa-response-audit-v1/README.md) confirms that the
fifteen cutoffs all hit exactly 65,536 generated tokens with no final answer,
well below the 256K context capacity. All count as incorrect in 163/198.
The natural-stop subset (163/183) is a diagnostic, not the full score.

The [native response audit](gpqa-audit-64k.json) matches all 198 private raw
responses to scored hashes, usage and finish reasons. One answer hit exactly
65,536 output tokens; none hit the model-context bound. The other 197 stopped
naturally, with 170 correct and 27 wrong. Removing truncation alone cannot
close the seven-answer gap to 177/198. The native control averaged 15.03 decode
tok/s/user, 473.42 aggregate output tok/s and 13.94 s TTFT for this variable
reasoning workload. Do not treat this as a fixed-context performance comparison.
Of the two matched 64K runs, 155 answers were correct in both, 15 only in the
native control, eight only in the candidate, and 20 in neither. A single sampled
run does not establish statistical significance or a numerical root cause.

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

The [native G0 receipt](native-g0/full-model.json) and
[hardware JUnit](native-g0/hardware.xml) now confirm a pass at 04:12:55 UTC,
after 43m18s of test execution including loading and warmup. All eight TP4
replicas produced matching tokens; concurrent/isolated TPOT ratios range from
0.99998 to 1.000064, within the unchanged 3% regression gate. This was a
63-token prompt with 128 output tokens and one user per replica, not a
long-context serving-throughput qualification. The JUnit copy adds only its
missing final newline. Full GPQA began through the standard eight-worker vLLM
endpoint at approximately 04:22 UTC and finished at 05:12:24 UTC with 170/198.
The unchanged GPQA gate failed. The [Tau3 pilot](../tau-pilot-v2/README.md)
subsequently finished at 3/12 successes in 32m36s, with four task timeouts and
one request timeout. It is not a matched published reference score. The
physical HTTP sweep is still running; no complete performance pass is claimed.

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

## Release packaging in parallel

TTIS branch `anatarajan/qwen38-galaxy-release-20261009`, commit
`e0e05bad5361d7c170068b3ad7b4df27de192250`, contains the experimental image
recipe, source/precision verifier, preparation CLI and standard TTIS ModelSpec
with a Galaxy Helm overlay. Twenty-one local wrapper/chart checks passed.
Both inference and init images must be digest-pinned before Helm renders.
No SJC3 cluster mutation or container hardware qualification has occurred.

The image's Qwen subtree is pinned to
`0abdc3403f039c46becef335ad02db99237593f8`: its complete runtime-source hash
set equals this G0 receipt. The latest model branch adds three independent
bandwidth-delivery probe files absent from the frozen serving snapshot, so
using its whole subtree correctly failed the strict source check. No check
was loosened. Compiled Metal remains `a08819ddbe23077f8037d3802303939064868ff6`
and the vLLM plugin remains `b7e4292e4193cba20abe9c7c68ce489201b2e36b`.

Image-only build work uses idle host `10.228.203.34`, a dedicated new directory,
and bounded container RAM. Attempt v1 stopped before compilation because the
host disables unprivileged user namespaces. Attempt v2 proved the isolated
mount-capable builder worked, but the client selected the rootless socket; it
was explicitly stopped and preserved. Attempt v3 fixed the client socket, then failed before any Dockerfile command
because nested cgroups were read-only. v4 verified a private cgroup remount
preserved the outer memory limit, but nested runc then failed its BPF device
query. Neither failure affected the model endpoint. v5 uses BuildKit rootless
spec conversion and an explicit rootless-cgroup runc wrapper inside the same
bounded, mount-capable container. Its tiny image probe (root write, chown and
UID-1000 access) passed before the full build reached native CMake configuration.
It ran as `qwen38-release-build-v5-20261009.service` with a four-hour limit,
24 CPUs and 192 GiB of container memory. No accelerators, host namespaces,
Docker socket or checkpoint are mounted into the builder. Only the new source
context and image-output directory are bound. Host kernel settings and existing
images are unchanged. Build caches use a container-local tmpfs; OCI output is
under `/dev/shm/qwen38-release-image-20261009-v5`. The completed archive was
subsequently copied to host disk with an fsync and matching full-file SHA-256.

The image build completed and its source/import verifiers passed. The
[image receipts](../image-build-v5/README.md) record its actual OCI digest,
archive checksum and durable host path. It is not yet registry-published,
container-hardware-qualified or deployed through Helm. The unchanged native
policy in that image scored 170/198 and is below the release gate.
[Build and Helm instructions](https://github.com/tenstorrent/tt-inference-server/blob/e0e05bad5361d7c170068b3ad7b4df27de192250/scripts/release/QWEN38_GALAXY.md).

A [higher-precision LM-head control](../accuracy-head-v1/README.md) is now
queued after the full current queue. It has its own frozen source, new G0 and
full GPQA; the current model endpoint and built image remain unchanged.
