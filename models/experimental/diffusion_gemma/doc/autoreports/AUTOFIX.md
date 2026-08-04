# AutoFix Report

## Starting Evidence

- GPQA smoke log: `/tmp/gpqa_fix4_server.log`.
- Both 256-token blocks completed, but `commit_latency_s` was 35.09 and 35.33 seconds.
- Source inspection verified that `generate._resolve_default_commit_fn` selected the 256-forward sequential commit whenever model-owned hybrid page tables were present.

## Hypothesis Experiments

- Hypothesis: one batched causal backbone pass can write the hybrid paged cache correctly.
  - Experiment: map a 32-aligned, 256-token canvas onto 64-token pages, preserving partial first/last pages and wrapping sliding-layer block IDs.
  - Result: exact device equality across a sliding-cache wrap (`start_pos=992`, block IDs `15,0,1,2,3`).
  - Verdict: verified.
  - Fix: added whole-page `paged_fill_cache` writes with offset staging.

- Hypothesis: sliding attention must read history before the bulk circular write.
  - Experiment: compare paged sliding commit attention with a torch causal+sliding oracle.
  - Result: PCC 0.999921.
  - Verdict: verified.
  - Fix: materialize the previous bounded window, attend `[history; canvas]`, then update the circular cache.

- Hypothesis: full-attention layers can use paged chunked SDPA after the bulk append.
  - Experiment: initial GPQA startup failed because a 32-aligned prompt is not necessarily 128-aligned, violating the chunked-SDPA start contract.
  - Result: refuted in its initial form.
  - Fix: front-pad Q to the previous 128-token boundary, retain 128-row Q/K chunks, and slice the synthetic rows from the output.
  - Verification: torch causal-attention PCC 0.999904 at `start_pos=32`.

## Final Status

- Fixed.
- GPQA two-sample smoke completed successfully on `bhqb`.
- Commit latency changed from about 35 seconds to 2.53 seconds on the first compile and 0.149 seconds once warm.
- Smoke artifact: `/tmp/gpqa_paged_batched_fix2`.
