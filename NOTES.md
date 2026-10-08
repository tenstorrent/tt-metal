# t274 — R1d: DiffVAE stage-5 NA reader (fewer/larger NOC transactions)

Base: origin/ttp/t48-ltx25-integrated 5833f56096f (decode 2.714 s, blx01 job 019).

## Findings so far (run 1)
- Stage 5 NA per chip: 1 head, head_dim 64 (2 tiles), gather 7x4x4 = 112 bricks, 8 tiles/kv chunk,
  14 kv chunks, 2 bricks/query chunk, per-brick mask, persistent, relative_table=false (stage 5 has a
  key phase under the H split, so EVERY mask tile is generated on device; it is skipped only when the
  chunk's WindowClamp equals the resident one). K/V ring mode 2, 3 columns.
- t263 job 020: math ablated (reader only) = 103.0 ms/block vs 120.8 default.
  ~308 work items/core x 14 kv chunks -> ~24 us (~32k cycles) per kv chunk. That is far more than
  ~8 DRAM + ~24 L1 NOC reads should cost, and BF8 (half the bytes) gave nothing.
- Hypothesis: per-brick mask generation (16 tiles/kv chunk, 512 word stores + bitmap each, when not
  skippable) dominates, not NOC issue. Unconfirmed.

## Diagnostic (run 1)
- Commits: a15c16b52b7 (cherry-pick of t263's DIFFVAE_NA_ABLATE reads|math), 53eb515183c (modes
  combine with '+', new `mask` mode skips per-brick mask generation). Both diagnostic only.
- blx01 build: $F/t263/b (t263 closed; reused, incremental) @53eb515183c. Driver/run scripts in t274-drv/.
- blx01 broker job 026: arms mm = math+mask, mr = math+reads, one process each, both profiled,
  -t 450. Output: /var/tmp/fasth3/t274/out_D (stage_tree_mm.txt, stage_tree_mr.txt, run.log).
  (Job 024 was a failed submit: the first driver skipped the checkout; exit 9, no device work.)

## Next step
- On wake: `ssh g15blx01 bash /var/tmp/fasth3/t274/drv/probe.sh` exits 0 -> read
  `grep neighborhood-sdpa out_D/stage_tree_{mm,mr}.txt` (the 1-count lines are stage 5 per block).
  mm << 103 -> mask generation is the cost: cache/skip it (e.g. make the WindowClamp key ignore the
  key phase, or a relative table with phase). mr << 103 -> K/V reads: brick-contiguous K/V.
