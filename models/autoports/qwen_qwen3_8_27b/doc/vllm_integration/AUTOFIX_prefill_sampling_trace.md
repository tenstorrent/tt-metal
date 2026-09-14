# AutoFix: packed-prefill sampling trace lifetime

Starting evidence: AUTODEBUG_prefill_sampling_trace.md and
packed_sampling_trace_failure.log identify persistent concat/tilize program
buffers surviving into the next decode replay. The allocation checker raises
before replay, so this failure is not dismissed as a harmless warning.

Repair: canonical generator.sample_prefill tracks input shape/dtype/layout
signatures for the current cache binding and releases old traces before the
first packing/sampling call for each new signature. Successful calls mark the
signature warm; repeated signatures keep traces. No alternative sampler exists.

Five host regressions cover first-use ordering, repeat retention, count/dtype
changes, cache rebinding and failed-sampling retry; a negative control removing
cache invalidation fails. Actual async serving with allocation tracking and
tracebacks enabled passes the original four-test shared smoke after the separate
host-output-boundary repair: reduced_sampling_final.log (4 passed, no skips).
Native concurrent S31/45/53/67 G70 following these host tests passes with exact
original native text (reduced_native_after_compat.json). Reduced final server log
has no remaining unsafe-allocation error and closes explicitly on normal exit.

Status: fixed on the reduced representative layers0/3 path. Final full-model
sampling and qualitative evidence must additionally pass before stage closure.
