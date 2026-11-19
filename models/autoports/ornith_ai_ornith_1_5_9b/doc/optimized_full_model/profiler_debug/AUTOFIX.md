# AutoFix: profiler collection

Contract: collect valid signposted full-model boundary evidence without changing
production arithmetic or suppressing profiler consistency checks.

The inspection-only AutoDebug report proves `rd_kick` and `pp_kdec` collide at
16-bit source ID1831. The failed invocation omitted the established path overlay.
The minimal repair is the existing byte-identical reader/header symlink overlay,
model-directory cwd (repository cwd otherwise takes precedence), and an absolute
fresh output directory. Set `TT_METAL_PROFILER_DIR` before importing Tracy to
match `-o`. No shared profiler source, kernel math or JIT cache was modified.

Discriminating results:

1. Original B32 reduced-generator profile fails at first drain with the exact
   source collision; profile_masked_before.json and its log retain the failure.
2. Same B32 workload with effective overlay completes device execution and its
   signposted replay, proving the source-ID repair. The broad B32 warmup produces
   excessive profiler metadata; host report processing separately fails with
   `Device data missing: Op22403072 ... trace_id=None`. This attempt is not
   accepted as a performance report. profile_masked_before_retry.json retains
   the workload result and logs/profile_masked_before_retry.log the failure.
3. Narrow the capture to the question being measured: reduced B1 complete-model
   boundaries, plus a separate real B32 recurrent guard-only before/after trace.
   This avoids duplicating the huge per-slot prefill preparation catalogue while
   full B32 correctness and quiet-host throughput remain independently measured.
   Exact final commands are in ../final_validation_jobs.json and ../commands.jsonl.

The failed raw captures remain at
`/home/ttuser/ornith-9b/artifacts/optimized_full_model_masked_before` and
`/home/ttuser/ornith-9b/artifacts/optimized_full_model_masked_before_retry`.
No reset or process kill was used. AutoDebug sandbox preflight failed before
child execution because unprivileged namespaces are unavailable; its authorized
inspection-only retry completed under `AUTODEBUG_SKIP_CHILD_SANDBOX=1`.

Final collection results are recorded in the parent profile/README.md and the
final command exit records. The B32 warmup/report limitation does not reduce the
advertised model batch/context contract. It limits that broad profiler fixture.
