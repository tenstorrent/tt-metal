# Work log

Date: 2026-07-30 UTC

- Started from branch `skillexp-cell/fuse-advise/phi`, HEAD `73e0b2d35eff`.
- Kept scope to fused decoder implementation, tests, and fused-decoder docs.
- Audited the complete functional topology and TTNN dedicated ops in source,
  bindings, model examples, and unit tests.
- Selected only the fused SiLU-multiply rewrite. Device equivalence is PCC 1.0
  against the functional decoder; prefill Torch-oracle PCC remains above
  0.999997.
- Preserved paged cache BF16 dtype, layouts, page size, deterministic semantics,
  LongRoPE behavior, and 131072 context contract. No capability reduction.
- Like-for-like A/B at batch 1/32 shows the final path beats traced functional
  decode at both batches. Prefill before/after is also recorded at both batches.
- First Tracy run used too many replays, overflowed device marker buffers, and
  failed report correlation. It was rejected rather than used. The fused-only
  three-sample rerun completed and generated `tracy/ops.csv` plus four
  `tt-perf-report` table/CSV families.
- Watcher and profiler were run separately. Watcher correctness passed six
  fused-path cases with no watcher/NoC error.
- The static dispatch test proves `FusedDecoder._mlp` owns the fused runtime
  method and contains no standalone `ttnn.silu`.
- Initial independent review returned `more-work-needed`: it rejected a
  one-run sub-percent latency claim, found a false no-layout-op sentence, and
  found a stale failing log citation.
- AutoDebug/AutoFix produced `AUTODEBUG.md`, `AUTOFIX.md`, and
  `layout_autofix.md`. The robust gate ran 40 independent processes and 40,000
  paired trace replays; both batches and both orders have mean/median 95% CIs
  strictly favoring fused. The layout audit maps the disputed operations to
  explicit width-96 Phi RoPE and records exact current-op contract blockers.
- README now cites the authoritative 9-pass correctness log, classifies the
  rejected log/profiler attempts, and records profiler teardown/category
  limitations.
- Fresh independent rereview verdict: `clean-pass`; see `stage_review.md`.
- Stage-owned commit SHA is appended after the checkpoint is created.
