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

## Next step
Read drv363.done, jobs.txt, res/*.json, res/run_ab_job*.log. If winners pass A/B: put them in
_BLOCKINGS (models/tt_dit/utils/conv3d.py ~486-496), commit, cherry-pick to -land, ttp push --detach.
If none win: report failed/done with the per-layer table, no table change.
Prior: #100 found T>1 slower (T in 3,5 only); T=2/4 untested until now.
