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

## Job 026 result (blx01, stage 5 NA ms/block)
- default 120.8; reader only (math ablated, t263 job 020) 103.0; mm (math+mask ablated) 42; mr (math+reads ablated) 95.2.
- So per-brick mask generation is the reader cost (~95 ms/block), not K/V NOC reads (~42). The #263
  "per-tile NOC issue" hypothesis is wrong for this op.
- Cause (inferred, not DPRINT-verified): W/H-edge cores (critical path) refill the 224-tile persistent
  mask block about 3x per W row: in W-fastest index order the first/last 2 chunks of every row have
  their own WindowClamp.

## Fix (run 2): DIFFVAE_NA_EDGE_ORDER=1 (opt-in), commit ac876509daa
- kernels/neighborhood_edge_order.hpp: EdgeGroupedOrder visits a core's range grouped by
  (head/batch, H edge position, W edge position), each group in index order. Reader and writer use it;
  compute is order-agnostic. Depth per axis = ceil((window/2) / (brick_sites * chunk_bricks)) = 2 for
  stage 5; T is not grouped. In the program hash. Output should be md5-identical to default (only the
  order changes); the arms differ in program hash, so check NA ms/block, not md5, for validity.
- Host check: the order is a permutation of each core's range (start 0/100/5000, 308 items).
- Not yet built or run on device: blx01 was unreachable (No route to host) at 2026-10-08 ~11:55 local.

## Next step
- When `ssh g15blx01 true` works and blx01.READY / broker fsm is healthy: `bash t274-drv/go274.sh`
  (copies bundle + scripts, starts driver: incremental build of $F/t263/b @ac876509daa, then ONE broker
  job, arms base / edge:DIFFVAE_NA_EDGE_ORDER=1, both profiled, -t 450, out /var/tmp/fasth3/t274/out_E).
  Hand off waiting on `ssh g15blx01 bash /var/tmp/fasth3/t274/drv/probe.sh`.
- Read: `grep neighborhood-sdpa out_E/stage_tree_{base,edge}.txt` (1-count lines = stage 5 per block),
  decode times in run.log, md5 of both arms. PCC/PSNR vs /var/tmp/fasth3/diffvae/ref: check how
  decode261.py scores (HOST_SEEDS / SARMS 4th arg of run274.sh; driver passes only 3 args today).
- Gain >= 5% (~0.14 s off 2.714 s) and quality neutral: flip default (edge_order_requested() default
  on, =0 off), cherry-pick ONLY ac876509daa + the flip onto a -land branch from
  origin/ttp/t48-ltx25-integrated (enum conflict: ablate_mask entry is absent there, resolve by
  dropping it), rebuild/verify, `ttp push --detach`; notes via `ttp push --own --detach`.

## Run 3 (2026-10-08 12:01 UTC)
- First go274 launch (11:58 UTC) refused at health stage: blx01 broker fsm=recovering after a chip
  PCIe drop/glx_reset (broker 031-041, ~11:48-11:58 UTC; hold ended 11:58 "ready for tenants";
  not during one of our jobs, chips not named in the status line).
- Relaunched: broker job 042 (base vs edge, ac876509daa, -t 450). Probe: `ssh g15blx01 bash /var/tmp/fasth3/t274/drv/probe.sh`.
- Next: read out_E as in "Next step" above.
