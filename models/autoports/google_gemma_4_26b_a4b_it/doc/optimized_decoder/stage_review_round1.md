# Independent stage review: round 1

Verdict: **MORE WORK NEEDED**

The fresh-context xhigh review found two blocking P1 issues:

1. Policy metadata and documentation claimed BF16/HiFi4 attention while the
   current-source profiler proved that inherited fused attention linears run
   BF16/HiFi2.
2. K-block 22 had only been paired with the known-unsafe one-core up/gate
   topology, so the existing geometry matrix did not prove K8/K11 faster than
   a safe larger-K candidate.

Remediation:

- Aligned `bfp8_experts_lofi` attention policy metadata and the context/docs
  with the profiler-verified BF16/HiFi2 runtime. Regenerated real-weight HF,
  paired sequence-1,024 performance, op profiler, device trace, watcher, and
  full-suite evidence against the changed source.
- Added safe `bfp8_geo_u4_d8_k22_22`. It passed both representative layers but
  lost to K8/K11: 1.340 vs 1.339 ms sliding and 1.539 vs 1.536 ms full over 21
  trace replays. Current `geometry_bfp8_*.json` contains the complete rows.

Round 2 rereview is required before completion.
