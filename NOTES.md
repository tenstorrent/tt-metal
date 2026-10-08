# t238 — DiffVAE next cut toward 1 s

## State (2026-10-07 23:52 UTC)
- Branch head: code commit 1a47d18ecb5 `diffvae stage 5: skip layout-only round trips behind DIFFVAE_S5_LEAN`
  on top of t48 76db90bbcfd. Opt-in, exact (no value change expected).
- Lever: on the keep-bricked 2-D stage-5 path, drop three untilize/tilize pairs per block:
  V's retile to heads, the tilize after the K/V halo exchange (key phase untilizes again), and the
  output untilize (out-proj retile tilizes again).
- blx01: build /var/tmp/fasth3/t238/b (Release, rc 0), checked out at 1a47d18ecb5.
- Driver: blx01 /var/tmp/fasth3/t238/drv/driver.sh, log drv/driver.log, marker drv/driver.marker.
  Broker job 869 (-t 600): arms def then lean (DIFFVAE_S5_LEAN=1), each warm-up + 2 timed seeds +
  deep profile (out/stage_tree_<arm>.txt) + host-noise seeds 0,1. Score: lean vs def md5, cmp238.py
  vs diffvae/ref -> out/cmp_<arm>.json.
- Baseline to beat: default decode 4.487/4.484 s (job 868), PSNR 55.1-55.8 dB vs ref.

## Next step
1. `ssh g15blx01 cat /var/tmp/fasth3/t238/drv/driver.{marker,log}`; read out/run.log DECODE lines,
   out/cmp_*.json and stage_tree_{def,lean}.txt.
2. If lean is faster and identical to def: commit "make DIFFVAE_S5_LEAN the default (=0 opts out)",
   cherry-pick code commits onto a -land branch from origin/ttp/t48-ltx25-integrated, `ttp push --detach`.
3. If the def arm timed out on cold JIT: rerun the job (cache is warm now).
4. Rank next levers from the def deep tree.

## Job 869 result (2026-10-08 light wake)
- def: 4.488/4.490 s (mean 4.489), host-noise PCC 0.99996, PSNR 55.6/55.2 dB vs ref. Deep tree: out/stage_tree_def.txt.
- lean: CRASHED after 64 s, TT_THROW storage.cpp:164 (run.log ~line 3672-3713). No timing, no output.
- Next: debug the lean path : "Tensor is not allocated" in neighborhood_attention.py:744 rephased() reshape of K after exchange_only — lean path deallocates the tensor it still reads, then rerun A/B.
