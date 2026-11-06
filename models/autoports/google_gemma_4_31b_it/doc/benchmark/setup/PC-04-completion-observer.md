# PC-04: optional full-phase completion observation (proposed, unauthorized)

Problem: stock plugin completion timestamps remain internal; adapter callbacks end before host-compatible sampling and final output construction. Neither polling nor existing request metrics supplies lossless work/timing correlation.

The patch adds disabled-by-default calls to an adapter-owned buffered observer, an immutable step ID on `TTModelInput`, scheduler-entry/built-input markers, a synchronous sampled-output marker, the actual async resolved-output marker (including final output), and a deferred decode state-application marker. It does not alter the plugin checkout, sampling policy, synchronization or profiler settings. Candidate source parses via `ast.parse`; no runtime or device validation occurred.

This is a concrete **hook proposal**, not a ready collector. Required implementation before activation:

- Observer allocates globally unique session/step IDs, immediately snapshots only CPU metadata, emits submission/completion/failure records without retaining mutable runner objects, and flushes a bounded buffer on explicit collect. Overflow fails evidence validation. Buffer/thread ordering must retain event timestamps even when state application races publication.
- Adapter records actual resident positions and reload decisions; scheduler positions alone can be stale. Built-input metadata contains IDs, chunk endpoints and slot mappings; no new device reads are allowed.
- Propagate the step ID into deferred prefill state records: current stock deferred-prefill `CompletedDecodeStep.model_input` is absent. The proposed state-applied marker by itself cannot correlate those records. Add explicit discard/failure outcomes and complete read-only phase partitioning, including idle gaps, before claiming full-phase coverage.
- The draft deliberately rejects lane-DP when enabled and does not support speculation. Confirm the selected Stage 10 launch is front-packed non-speculative; implement and validate other routes if required. Synchronous skipped/empty paths and errors must close observer records.
- Validate two overlapping async steps, replace-preserved IDs, intermediate/mixed final prefills, host sampling, final-step tail, deferred state application, failures and discarded outputs with mocked CPU-only tests. Then validate overhead and event completeness on actual supported profiles without profiling.

Authorization would need to cover these runtime/interface hooks and the corresponding adapter collector integration. PC-02/03 test-only approval does not authorize it. No runtime exception has been requested or granted. This proposal is incomplete and not ready to execute. No patch has been applied; current runtime remains stock. Existing benchmark latency evidence may be retained, but this proposal supplies no measured roofline values.

Review correction: the draft preserves the original `@torch.no_grad()` decorator on `execute_model`; the observer is undecorated. AST assertions verified both conditions, all three candidate modules parse, and `git apply --check` succeeds. These checks do not validate runtime behavior.
