# t363 conv VAE per-layer conv3d blocking sweep — notes

Branch ttp/t363-conv-vae-per-layer-conv3d-blocking-sweep (base origin/ttp/t48-ltx25-integrated ba3636df5ee).
Code commit: a7a8a4762a3 (adds s1_res to the LTX-2.5 halo sweep list).

## Running (started 2026-10-10 11:54 PDT)
blx01 driver /var/tmp/fasth3/t363/drv363.sh (copy in tt-project/t363/), detached via ttp detach:
retry_when: ttp detach --check --host g15blx01 /var/tmp/fasth3/t363/detach/drv363
Steps: build t363/b (reflink t48 @ ba3636df5ee + copied test files) -> 4 broker sweep jobs
(s2_res s3_res s3_chg s1_res; 12 T=2/4 blockings that fit L1 + table seed; -t 600 each) -> pick363.py
(winner = >=3% faster than table) -> if any, one A/B full 4x8 decode job (test_vae_ltx_blk_ab_4x8.py,
real 2.5 conv VAE, seed0 latent; arm A table, arm B patched; bit-identity/PCC/PSNR) -> delete jit, b.
Marker /var/tmp/fasth3/t363/drv363.done; job list t363/jobs.txt; results t363/res/ (sweep JSONs,
run_<name>_job<ID>.log, winners.txt).

## Sweep results (jobs 457 s2_res, 462 s3_res, 463 s3_chg, 465 s1_res; blx01, 900 MHz clamp, relative)
Blocking = (Cin_blk, Cout_blk, T, H, W); halo sweep on create_submesh(2,4) of the opened 4x8.
| layer  | key (Cin,Cout,T,H,W)      | old (table)     | old us | new             | new us | delta | output |
| s1_res | 512,512,39,17,15          | 64,256,1,4,8    | 2863   | 64,256,2,2,8    | 2435   | -15%  | md5 identical |
| s2_res | 512,512,75,34,30          | 64,256,1,8,4    | 17343  | 64,256,2,4,4    | 16319  | -6%   | md5 identical |
| s3_res | 256,256,147,34,30         | 64,256,1,8,4    | 7481   | 64,256,2,4,4    | 6372   | -15%  | md5 identical |
| s3_chg | 256,512,147,34,30         | 64,256,1,8,4    | 14469  | 64,256,2,4,4    | 11904  | -18%  | md5 identical |
Cin_blk 32 and Cout_blk 128 variants were all 1.4-2.5x slower; T=4 never beat T=2.
Code commit ced1ac7f2bd: table entries + CPU test + sweep candidates gain T=2/4.

## A/B attempts
- 468 (12:18 PDT): failed in the test's _latent (seed0.pt is a BCTHW tensor, not a dict). Fixed.
- 473: ran on a stale tree (a killed first launch left a t48 copy at bf7db12a14 with a _ttnn.so, so the
  build was skipped); ImportError vae_key_map. Driver now stamps finished builds (.t363_built).
- drv363c (started 12:26 PDT): drv363b.sh, builds t48 tip 20b40f459a9 (fetch), A/B job, cleanup.
  retry_when: ttp detach --check --host g15blx01 /var/tmp/fasth3/t363/detach/drv363c
  Marker /var/tmp/fasth3/t363/drv363b.done, log drv363b.log, result res/run_ab_job*.log ("AB RESULT").

- drv363c: build failed (19:31 UTC) at 1069/1445 linking tracy-capture: ld.lld 'pthread_create has failed:
  Resource temporarily unavailable' (transient host thread limit, not a code error). No device job ran.
- drv363d (started 12:40 PDT): same driver, build capped at CMAKE_BUILD_PARALLEL_LEVEL=32.
  retry_when: ttp detach --check --host g15blx01 /var/tmp/fasth3/t363/detach/drv363d

- drv363d: build ok (19:41 UTC); A/B broker job 479 (19:41-19:44 UTC, blx01 full 4x8, 900 MHz clamp, relative):
  A_old yuv decode [0.5893, 0.6077, 0.5971] median 0.5971 s; B_new [0.541, 0.555, 0.5571] median 0.5550 s;
  new/old 0.929 (-7.1%); identical=True pcc=1.000000 psnr=inf maxabs=0. jit and b cleaned. No drops.
  Summary: tt-project/t363/res/ab_job479_summary.txt.

## Next step
Land ced1ac7f2bd on ttp/t48-ltx25-integrated (cherry-pick onto a -land branch, ttp push --detach).
