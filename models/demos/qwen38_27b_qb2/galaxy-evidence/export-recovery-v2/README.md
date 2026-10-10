# Bounded export recovery for the compact full-model profile

The queued compact qualification/profile uses the installed Tracy exporter,
which exports every CPU child zone before filtering the result. The previous
full-model capture exceeded its 8-GiB file limit after the hardware test had
already passed. This is a reporting failure, not an inference failure.

The new CPU-only recovery copies the completed trace and compact device CSV to
a fresh directory, filters TT operation timing at export, retains all messages
and signposts, and reruns the existing complete-model reconciliation. Optional
host child-function timing is omitted. Device timings are retained. The original
capture, failed receipt, native installation and active source remain unchanged.

Acceptance requires passing XML and clean receipts for both profiled and
unprofiled hardware tests, identical source/input/output hashes and precision,
all four ranks, three replays and all 64 layers. Recovery runs with a 2-GiB
per-file limit, an 8-GiB total limit, at least 16 GiB free and a 15-minute bound
for each export/processing child. The user service is limited to 16 GiB and four
CPU cores; no physical devices are accessed by the export process.

The persistent watcher waits for the exact compact-followup invocation to exit.
On normal success it does nothing. On export failure it requires completed GPQA
and worker cleanup, a passing CPU recovery validation, passing hardware capture
receipts and successful report reconciliation. Only then may it retry the five
existing followers, in order, using fresh units/directories and unchanged frozen
sources. Every original follower must have terminated without starting hardware
and must report precisely an unsuccessful predecessor. A missing unit, changed
invocation, hardware failure or partial report cannot authorize a retry.

The launch order remains projection tuning, prefill attention batching, compact
GDN long-horizon validation, register-resident recurrence, compact gates. The
first replacement waits for the watcher to exit cleanly before acquiring the
existing hardware lock. No serving policy is promoted. All services survive an
SSH/session disconnect, not a host reboot.

CPU preflight: **595 passed, one skipped, 91 subtests passed**. The earlier local
attempt to run existing profile tests without the repository conftest missed
the `expect_error` fixture; the complete frozen host suite used the correct
fixtures and passed. Export v1 rejected a historical receipt that predates the
top-level expected-recurrence field. V2 accepts that schema only when both
effective precision policies match, and uses streaming hashing compatible with
the host's Python 3.10. V1's unused watcher was stopped; its evidence is retained.

The captured control, unit, source hash and validation receipts in this directory
record the live state at collection time. Kernel performance remains **20.035
TSU at B16/32K/TP4** and **22.751 at 16K**; this tooling adds no claimed model
speedup.

Validation finished at **21:13:24 UTC**, in **109.2 seconds**, using the previous
complete P0 capture without accessing hardware. The filtered timing CSV is
11,827,326 bytes (the failed original export exceeded 8 GiB). All rank/replay
statistics and host/device comparisons exactly match the previously recovered
report; host/device gaps remain **0.403%-0.461%**. The live fallback is
`qwen38-compact-export-fallback-v2-20261010.service`, invocation
`e37b2f244e924e2f9b03f58af90d47ba`. At collection it was waiting for the original
compact GPQA/profile controller, which remained active with unchanged invocation.
GPQA was **189/198 completed, 175 correct, zero truncations**; this is partial.
