# Persistent higher-precision LM-head control

Queued on `10.228.203.98` at 05:17:41 UTC, Oct 9, 2026, in user service
`qwen38-accuracy-head-v1-20261009.service`. It was confirmed live and waiting
on the exact invocation of `qwen38-overnight-v2-20261009.service`. No model or
accuracy result is claimed for this new policy yet.

The controller waits for a terminal predecessor service, all three completed
hardware stages and confirmed serving-worker cleanup. A completed low GPQA
score allows the next experiment; a failed device stage or incomplete cleanup
does not. A live owner, stale receipt or changed service invocation cannot
authorize reuse. Transient observation failures keep waiting. The normal G0
and serving runners still acquire `/tmp/tt-device.lock`.

The experiment changes only the LM head to BFP8/HiFi2. Native recurrence,
accurate full-tile attention, all 64 decoder policies, checkpoint and sampling
stay fixed. It first runs a new eight-replica G0, then all 198 GPQA questions
with 65,536 output tokens, T1/p.95/k20, seed 42 and concurrency 128. Raw responses
remain private on the host. The release gate is still 177/198.

The source snapshot is `accuracy-head-source-v1`; eventual results will be
`accuracy-head-v1`, both beneath `/home/ttuser/qwen38-artifacts-20261007`.
The separate control directory is `accuracy-head-control-v1`. Source hashes
are checked before hardware begins. The service has an 18-hour total bound,
a 12-hour predecessor-wait bound, 256 GiB host-memory cap and 16 CPU quota.
It survives client disconnect, but does not automatically resume after reboot.
Cancellation affects only the named follow-up service.

Preflight: **381 tests and 40 subtests passed**; one optional real-tokenizer
test initially skipped because the staging environment omitted
`MODEL_WEIGHTS_DIR`. The same test was run separately with the pinned local
tokenizer; its receipt is retained alongside the full preflight. The normal
hardware controller sets that variable and repeats preflight before G0.
The initial staging attempt failed before tests because `LD_LIBRARY_PATH`
omitted the existing native libraries; its log is preserved. Correcting that
environment did not change native installations or the running queue.

Six new tests cover service ownership, changed invocations, missing/stale
receipts, unsuccessful cleanup and propagation of the new precision through
G0 and full GPQA while skipping unrelated experiment stages.

Use `systemctl --user show qwen38-accuracy-head-v1-20261009.service` plus the
control directory's `status.json` for current status. The committed service
and status files are launch snapshots, not claims that the experiment passed.

The published tokenizer JUnit adds only the missing final newline.
