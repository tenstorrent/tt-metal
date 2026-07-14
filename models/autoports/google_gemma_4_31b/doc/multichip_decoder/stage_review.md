# Stage 04 independent review

Verdict: `clean-pass`

Required work: none.

The fresh rereview inspected the final TP4 implementation and tests, context
and physical-memory calculations, all JUnit and watcher evidence, runtime
fallback audit, profiler provenance, and the final Tracy/`tt-perf-report`
windows. It verified that the prior required work is closed:

- The final 24-core square-MLP decode geometry is the fastest PCC-clean option
  in the 8/12/21/24-core full traced-layer sweep.
- The final short-prefill 24-core 1-D geometry is the fastest PCC-clean legal
  option; DRAM-sharded prefill has a retained exact `M == 1` blocker.
- The adapted fractured boundary passes through distributed norms, residual
  addition, delayed gather, and the real next gate projection, but is slower
  than the retained replicated boundary in decode and prefill.
- Worker and NoC watcher coverage is clean; the documented Ethernet-only
  instrumentation limitation does not affect normal decoder execution.

No hard-check gaps or blocking concerns remain. The reviewer concluded that
the implementation is a real, trace-safe TP4 decoder suitable as the next
stage's layer-stack baseline. Full-model execution remains intentionally out
of scope for Stage 04.
