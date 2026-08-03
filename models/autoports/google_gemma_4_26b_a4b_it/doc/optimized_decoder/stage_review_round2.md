# Independent stage review: round 2

Verdict: **MORE WORK NEEDED**

The rereviewer verified that both round-one findings were fixed, then found
one new blocking P1: dominant prefill sparse up/gate and down remained on the
fused K1 configuration even though they represented 65-66% and 19% of the
prefill profile. Decode-only geometry did not satisfy independent prefill
tuning.

Remediation:

- Added an owned optimized `_moe_prefill_tile` using BFP8/LoFi expert compute
  and phase-specific sparse program configs.
- Added `test_optimized_sparse_prefill_geometry`, which compares sequence-1,024
  4/8-core K1/K1, 2/4-core K4/K2, 2/4-core K11/K11, 4/8-core K8/K11, and
  4/8-core K22/K22 for both representative layer kinds.
- Selected 4/8-core K8/K11 independently for prefill. It reduces warmed
  prefill from 400.921 to 239.641 ms sliding and 401.868 to 240.887 ms full,
  while decode retains its separate K8/K11 selection.

A fresh clean-pass rereview is still required.
