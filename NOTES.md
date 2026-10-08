# t262: R3 fused DiffVAE stage-5 block (non-NA part)

## Done (run 1040, 2026-10-08)
- Code commit e0c550083cd on ttp/t262-r3-fused-diffvae-stage-5-block-non-na-pa (base t48 @be5d1bc045a), Python only:
  DIFFVAE_S5_FOLD_ADDS=1 (opt-in, default off). Context inject, attention residual and MLP residual adds run in
  the epilogue of the feeding matmul (dit_minimal_matmul_addcmul_fused, ones row as b); the out-proj only computes
  the interior rows (crop before proj). Linear.forward gained addcmul_a/addcmul_b; SwiGLU.forward gained residual=.
  Tests: unit test_fold_adds_default_off; gate test_stage5_parity_w_sharded_bricked parametrized fold_adds.
  Not run on device yet. ttp checks pass.
- Estimate: ~7-8 ms/block saved (~50-60 ms decode). Far below the -0.4 s accept bar on its own.
- A/B scaffolding: tt-project/t262/ (copied from t253: run262.sh, decode262.py, driver262.sh, cmp241.py, env.yaml).
  run262.sh puts $T/src/$SRC first on PYTHONPATH over the t238/b build (34a571c5f47 C++; no C++ change here).

## Blocker
- g15blx01 "No route to host" (ssh + ping) since ~04:50 UTC 2026-10-08. blx03 has no DiffVAE setup/refs;
  g15blx02 device paused (#176).

## Next step (when ssh g15blx01 works)
1. Health: ssh g15blx01 'tt-device-mcp status'; t238/b still at 34a571c5f47 (driver refuses otherwise).
2. mkdir /var/tmp/fasth3/t262/{drv,src/e0c550083cd}; git archive e0c550083cd models | ssh g15blx01 tar -x -C /var/tmp/fasth3/t262/src/e0c550083cd;
   scp tt-project/t262/* to drv (tmp name + mv).
3. ssh g15blx01 'setsid nohup bash /var/tmp/fasth3/t262/drv/driver262.sh AB e0c550083cd "def fold:DIFFVAE_S5_FOLD_ADDS=1" "def fold" 330
   > /var/tmp/fasth3/t262/drv/nohup.log 2>&1 &'  (t253's 2-arm job with one profiled arm fit 330 s; if both profiled
   arms exceed it, profile only fold and reuse t253's def tree from job 912).
4. Wait on: ssh g15blx01 test -e /var/tmp/fasth3/t262/drv/driver_AB.marker. Read driver.log, out_AB/cmp_*.json,
   stage_tree_*.txt. Pass: PCC >= 0.9999, PSNR within 0.5 dB of 55.04/54.59 dB.
5. Remaining R3 sub-steps (not started): (c) packed-output flag on dit_fused_distributed_rmsnorm (writer CT arg
   head_dim_tiles = num_tile_cols -> row-major packed; needs C++ rebuild in an own build dir under /var/tmp/fasth3),
   est. -140 ms; (a) K+V one halo CCL, uncertain; (b) SwiGLU hidden in L1, est. ~4 ms/block, marginal.
   Even all together the estimate is ~-0.2 s, below -0.4 s: do not land unless the measured gain clears the bar.

## Result (blx01 job 947, 2026-10-08 05:05-05:10 UTC, -t 330, rc 0, no drops)
- Warm decode, 2 seeds, one arm per process: def 3.374 s, fold (DIFFVAE_S5_FOLD_ADDS=1) 3.374 s. No gain.
- Profiled stage 5: 2398.7 -> 2387.7 ms (-11 ms, ~1.4 ms/block). Residual crop+add (2.7 ms) is gone but the
  out-proj with the folded epilogue grew 2.7 -> 4.7 ms; context-inject 4.8 -> 4.6 ms.
- Quality vs #214 refs: def PCC 0.99995/0.99995, PSNR 55.04/54.59 dB; fold PCC 0.99995/0.99995,
  PSNR 54.97/54.51 dB (-0.08 dB). fold vs def 55.7/55.3 dB. md5 differ (valid A/B).
- Stage-5 non-NA per block (def): qkv-proj 5.6, qkv-lanes 27.8, halo+brick 42.4 (k/v halo 10.4 each,
  rebrick 3.1+3.0, untilize 1.8+1.8, unattributed 11.8), out-proj 2.7, context 4.8, norm+mod 3.3+2.6,
  residual 2.7, mlp 25.6.
- Evidence: tt-project/t262/out_AB/ (stage trees, cmp json, driver.log).

## Decision
- DIFFVAE_S5_FOLD_ADDS rejected: stays opt-in, default off, not landed on t48 (code commit e0c550083cd on this
  branch only).
- R3 stopped. The remaining levers ((c) packed norm output, est. -140 ms; (a) one K+V halo CCL, uncertain;
  (b) SwiGLU hidden in L1, ~-30 ms) add up to ~-0.2 s even if all estimates hold, below the -0.4 s bar and
  each below the 5% single-lever cut (#257). The biggest single item is the K/V halo+brick (42 ms/block,
  ~0.34 s over 8 blocks), which belongs with R1 (NA K/V L1 ring) rather than here.
