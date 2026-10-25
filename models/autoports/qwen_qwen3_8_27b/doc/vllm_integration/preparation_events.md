# Preparation timing evidence

`Generator.preparation_events` is an append-only list populated only by
`prepare_decode`. A second call after preparation preserves the list unchanged;
a retry after failure appends another attempt. Adapter warmup can export the list
as its baseline for subsequent request/trace events.

Each record contains `attempt`, `phase`, `signature`, `trace_id`, `status`,
`attempt_status`, `host_start_ms`, and `host_elapsed_ms`. Signatures identify batch
and operation kind; prefill includes physical length, warmed start alignments,
logits modes and participating layer indices. Decode records masked/unmasked
variants; sampling records the canonical strategy and buffered/unbuffered variant.
Capture IDs are assigned from the actual `begin_trace_capture` result. A completed
capture is recorded only after `end_trace_capture` succeeds.

Times are host preparation/JIT/dispatch and capture API durations. They are not
device execution times or isolated compiler times. No synchronization, profiler,
per-token logging, or new environment option was added. Existing synchronization
remains part of whole-attempt preparation duration.

Failed attempts retain completed phase records, mark the active failed phase,
and mark every record's `attempt_status=failed`. Cleanup release events record
actual release outcomes separately. Whole-attempt summaries count completed
captures and completed cleanup releases directly from those events, including
captures completed before a later failure.

Validation: container `/opt/venv/bin/python -m py_compile` passes. A host-only
AST extraction of `prepare_decode`, using fake tensor/trace operations, verifies
successful signatures/IDs, bucket timing, idempotence, partial-capture failure,
cleanup counts, and append-only retry behavior. Compact results:
`readiness_vllm/preparation_events_host_check.json`. No devices were opened for
this check; representative serving execution remains the parent stage's gate.
