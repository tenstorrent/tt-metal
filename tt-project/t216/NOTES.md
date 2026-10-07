# t216: DiffVAE stage-5 GNA stride (2,4,4) vs 1 on 4x8, 5 seeds

Code: branch ttp/t216-gna-stride (worktree tt-project/worktrees/t216), 709b3bad69a on t48 a40d78b8bae:
DIFFVAE_GNA_STRIDE=t,h,w in DiffVAEOptions.production(), default 1,1,1. Not pushed yet.

Device job (blx01, /var/tmp/fasth3/t216, scripts here are the deployed copies):
- driver216.sh (setsid, started 2026-10-07 19:48 UTC) -> broker job 830 (-t 600, unmeasured):
  run216.sh -> ab216.py on the t212 port build /var/tmp/fasth3/t212/b (8b1167ef43b, tree-identical
  to t48 a40d78b8bae). One process, arms 1x1x1 then 2x4x4 (options = production() with gna_stride
  replaced), #214 latents + #214 host noise, warm-up then seeds 0-4. Per seed: total decode time and
  stage5.forward time (sync-bracketed; includes host x_t embed and pixel pull).
- Then score216.py (CPU) -> out/score.log, out/scores.json: PSNR/PCC 2x4x4 vs 1, each vs #214 ref,
  floor, seam ratio (16 px / 2 frames), stills still_*_f72.png, diff8x_*, crop512_AB_*; mp4s.
- Marker: driver.marker "T216_DRIVER_DONE stage=.. rc=.. jobs=..". Log: driver.log, out/run.*.log.

Next on wake: read marker, out/run.*.log ("[t216] DECODE"/"MEAN"), out/score.log; copy scores,
stills and logs to tt-project/t216/results; view stills/crops; decide; if accepted, push knob to t48
(ttp push); delete out/s2x4x4_*.yuv on blx01 (keep 1x1x1 yuv as ported-build reference).
