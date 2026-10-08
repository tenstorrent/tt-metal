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

## Job 044 result: INVALID A/B (dr path never ran)
- def 2.311-2.317 s, dr 2.310-2.313 s, md5 identical per seed, stage-5 tree identical (k/v untilize +
  key-phase rebrick present, no tiles-to-sticks scope). Halo+brick 42 ms/block in both.
- Cause: stage 5 packs 4 heads per lane, channels=256 (16 KB stick, _halo_split=4); guard needed
  channels==64 and parts==1.
- Fix: code commit 2c8539570d7 (kernel takes CHANNELS and PARTS as compile args; mode 0 writes the
  exchange's split shape directly; mode 1 page = s / (32/PARTS); layout guard per K/V tensor).

## Job 045 (blx01, 2026-10-08 12:25 UTC, -t 340): rerun A/B with 2c8539570d7
- Overlay $T/src now = git archive models/ @2c8539570d7 (REV file). Out /var/tmp/fasth3/t276/out_B.
- Next: confirm dr tree has "tiles-to-sticks"; md5 dr == def per seed; decode means; ms/block;
  score: $F/t48/python_env/bin/python $F/t276/drv/cmp241.py $F/diffvae/ref $F/t276/out_B/dr $F/t276/out_B/cmp_dr.json 0,1,2,3,4
- Bar: gain >= 0.116 s (5% of 2.31 s) -> flip default + land on t48; else notes branch.

## Job 045 result: dr arm crashed (def arm fine: 2.309-2.312 s)
- TT_FATAL input_tensor.is_allocated() in neighbor_pad_async inside the k exchange (warm-up seed 0).
- Cause: direct_rephased deallocated `exchanged`, the ccl manager's cached zero-padded ping-pong
  buffer (get_np_ping_pong_buffer), which later neighbor_pad calls reuse. Fix: code commit d1d466afe4b.

## Job 049 (blx01, 2026-10-08 12:30 UTC, -t 340): rerun A/B with d1d466afe4b
- Submit: tt-device-mcp run-bg -w $T -e $T/drv/env.yaml -t 340 "bash $T/drv/run276.sh \"def: dr:DIFFVAE_NA_DIRECT_REPHASE=1\" $T/out_C \"def dr\" \"dr\""
- Overlay $T/src = git archive models/ @d1d466afe4b. Out /var/tmp/fasth3/t276/out_C.
- Next: same checks as job 045 (tiles-to-sticks in dr tree, md5 dr == def, decode means, ms/block, cmp241 on out_C/dr).

## Job 049 result (blx01, valid A/B, one arm per process, no drops)
- dr path ran (tree has tiles-to-sticks scope); md5 dr == def on all 5 seeds (quality-neutral).
- Decode 1080p 145f 4x8: def 2.310-2.320 s (mean 2.314), dr 2.236-2.243 s (mean 2.240): -74 ms, -3.2%.
- PCC/PSNR vs #214 ref (dr, same as def): 0.99995/54.71, 0.99995/54.28, 0.99995/54.25, 0.99995/54.86, 0.99995/54.50 dB.
- Deep-profile trees (seed 0, one run) show dr halo+brick higher in some stage-5 groups (58 vs 31 ms);
  deep profile is not reliable here (sync per scope, generic_op first-call); timed decode is the metric.
- Verdict: below the 5% bar (0.116 s). Stays opt-in (DIFFVAE_NA_DIRECT_REPHASE=1), not landed on t48.
  Remaining copy cost is the NP-H/NP-W exchange itself plus mode-0/mode-1 passes; a further step
  would gather from band+halo without the stick round trip (full edge-slab design).
