# Stage 02 independent review

## First review

Verdict: `more-work-needed`.

Required work: the initial topology audit had rejected shared-LHS projection
packing without an adapted implementation/measurement. The reviewer also found
that cache and real-weight artifacts did not independently prove fallback
hard-failure, and watcher artifact wording was too broad.

## Remediation

- Packed full-attention Q+gate/K/V and linear-attention QKV/Z/beta/decay
  setup-time weights; runtime uses one projection and exact slices.
- Re-ran PCC, non-aligned prefill, trace b1/b32, profiler b1/b32, paged-cache
  routing, real-weight decode, and watcher b32 gates.
- Final profiler totals beat all corresponding baselines.
- Cache and real-weight runners now set and print
  `throw_exception_on_fallback=true`.
- Watcher documentation points to the actual final console/generated logs.

## Fresh rereview

Verdict: `clean-pass`.

Required work: none. The reviewer independently re-derived all eight final
profiler totals, verified both packed projection families and exact slicing,
and found the final PCC, cache, fallback, trace, and watcher evidence closed
the prior findings. Controlled teardown-only anomalies were TTNN's initial
pre-run config print and post-success nanobind leak diagnostics.
