# AutoFix remediation

The independent stage review returned `more-work-needed`, so the `$autofix`
workflow began with a fresh-context AutoDebug inspection. AutoDebug made three
source-backed findings: optimized prefill never reached the exact sparse
kernels and consumed about 96.5% of the seq-256 device window; per-role and
per-batch DRAM QKV/packed/down evidence was absent; and the dedicated full
profiler used shared-cache geometry instead of the natural full-attention
geometry.

Three isolated repairs were tested before integration:

1. An optimized `_moe_prefill_chunk` path exposed exact grid, K-block,
   placement, and internal chunk-size candidates without changing public
   padding/slicing. The selected 11x2/11x2/11x8, w11, L1, chunk32 policy
   reduces seq-256 from 168.959 to 32.093 ms.
2. Per-role BF16 DRAM-sharded QKV, packed-dense, and dense-down candidates
   were added for batch 1 and 32. Layer-specific batch-1 QKV and batch-32
   packed-dense are proven winners; correctness failures and regressions are
   retained rather than hidden.
3. Dedicated profiler geometry now follows the real cache contract: sliding
   is shared and full is natural/non-shared. Runtime invocation counters and
   same-run host/device accounting were added.

The first integrated batch-32 QKV candidate was fast but failed full-attention
PCC at 0.984152 and was removed. A global batch-1 QKV width-2 policy passed
trace-only checks but failed the stricter sliding real-weight case at 0.993311;
it was repaired by selecting width 1 only for sliding and width 2 only for
full attention. Frozen gates then passed: correctness 14/14, stress 4/4,
performance 4/4, batch-32 prefill 4/4, watcher 9/9, and default suite 23 pass
plus 18 opt-in skips.

The original root `AUTODEBUG.md` was a transient workflow report. Its direct
observations, hypotheses, experiments, and dispositions are preserved here,
in `candidate_matrix.json`, and in `work_log.md`.

The first post-AutoFix review identified two remaining topology gaps and two
evidence gaps. The packed sparse projection was then implemented and selected
(32.093236 -> 21.490080 ms at seq256). Batch-32 QKV was specialized by layer
kind; sliding w2 is correct/faster, while full retains the packed-only path due
to its 0.984152 QKV PCC. Final performance JSONs were regenerated last, and
`profiler_accounting.md` now provides bytes/512-GB/s roofline time plus the
required device conversion ledger. A fresh rereview is required after these
repairs.
