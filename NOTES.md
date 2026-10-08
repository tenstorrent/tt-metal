# t276 notes: direct key-phase K/V for DiffVAE stage-5 NA

## Change (code commit ce1286834c6 on origin/t48 a5a774ea17f, opt-in DIFFVAE_NA_DIRECT_REPHASE=1)
Per lane: generic_op kernel (models/tt_dit/layers/kernels/na_brick_gather.cpp, JIT, no C++ build)
mode 0 tiled band -> 4 KB brick sticks; existing NP-H + NP-W exchange; mode 1 gathers the key-phase
tiles from the exchanged sticks with the existing _cached_gather_index table. Drops untilize,
the C->32C and 32C->C row-major reshape copies and the embedding gather. Expected md5-identical.
Simpler than the full edge-slab design (NP volume unchanged); est. ~3.6 of 8.8 band-copy units/lane saved.

## Device run (blx01)
- Overlay /var/tmp/fasth3/t276/src (git archive models/ @ce1286834c6) on C++ build t263/b @a5a774ea17f.
- Job 044 (2026-10-08 12:15 UTC, -t 560): arms def, dr; seeds 0-4 timed; both deep-profiled; dr host seeds 0-4.
  Out /var/tmp/fasth3/t276/out_A (run.log, stage_tree_*.txt, dr/ref_dvx_seed*.yuv).
- Score: F=/var/tmp/fasth3; $F/t48/python_env/bin/python $F/t276/drv/cmp241.py $F/diffvae/ref $F/t276/out_A/dr $F/t276/out_A/cmp_dr.json 0,1,2,3,4
- Next: check md5 dr == def per seed, decode means, stage-5 halo+brick ms/block; if gain >= 5% of decode, flip default, land.
