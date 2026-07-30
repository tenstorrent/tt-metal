# AutoFix report

## Starting evidence

- Fresh `AUTODEBUG.md` was run after the first independent stage review.
- Original findings: the single A/B traced-decode win was too small to call
  robust, and all bounded profiler CSVs contradicted README's claim of no
  tilize/untilize operations.

## Hypothesis experiments

- Hypothesis: the SiLU-multiply end-to-end win is order or host-timing noise.
  Experiment: equal-warmup interleaved paired functional/fused trace timing,
  both candidate orders, batch 1/32, 10 independent processes x 1000 pairs.
  Result: all four paired mean and median hierarchical-bootstrap 95% CIs are
  below zero; mean improvement is 2.44-2.82 microseconds.
  Verdict: refuted; the win is small but reproducible.
  Evidence: `repeated_ab/analysis.txt`, 40 JSON files, and 40 process logs.
  Fix: benchmark now retains samples and supports order/batch controls; analyzer
  computes deterministic hierarchical-bootstrap CIs.
  Verification: the full 40,000-pair command completed successfully.

- Hypothesis: norm, head, cache, mask, or avoidable layout boundaries emit the
  profiler's tilize/untilize/permute families.
  Experiment: source-contract audit plus per-call ordered CSV adjacency.
  Result: all named families map to two explicit Q/K width-96 Phi RoPE groups;
  B1 decode has two additional embedding-output padding conversions. Other
  proposed sources were refuted. Generic rotary requires width 32 or a multiple
  of 64; Llama rotary's one-tile transform cannot implement midpoint-48 Phi
  rotation.
  Verdict: verified attribution; current TTNN op set has an exact blocker.
  Evidence: `layout_autofix.md` and four bounded report CSV families.
  Fix: corrected the docs; no speculative implementation change was kept.
  Verification: existing fused PCC, trace, watcher, and profiler evidence is
  unchanged because the runtime did not change.

## Final status

Fixed. Performance is reproducible, layout movement is accurately attributed,
and every remaining material movement belongs to a Phi RoPE operation not
expressible by an available dedicated kernel. Remaining risk is that a future
Phi-specific TTNN RoPE kernel could remove that family.
