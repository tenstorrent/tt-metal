# t264 notes: R1c DiffVAE stage-5 K/V halo + rebrick

## Baseline profile (blx01 job 019, t261/out_R/stage_tree_def.txt, t48 @5833f56096f, decode 2.714 s)
Stage-5 block 0, halo+brick-permute (k,v) 42.3 ms:
k untilize 1.8, k halo-exchange 10.4, k key-phase rebrick 3.1, same for v (1.8/10.3/3.0), unattributed 11.8.

## Finding (code read)
Per lane the band is copied ~7 times: untilize, reshape C->32C (reshape_rm copy: an RM reshape that
changes the last dim is never a view, reshape.cpp this_is_view), reshape 32C->32C/4 (copy), H neighbor_pad
(full local copy), W neighbor_pad (full local copy), reshape ->C (copy, the "unattributed"), embedding gather.
neighbor_pad's local copy moves one stick per NOC read barrier, so 512 B sticks are likely slow.

## Change (bb6b2c20e69, opt-in)
- DIFFVAE_NA_HALO_2D=1: one fused 2-D neighbor_pad (dims [2,3], H axis + W axis, phase 2 reads the W halo
  from the H-padded output, so corners match) and the untilized rows reshaped once straight to the 4 KB
  sub-column sticks. Removes 2 of the 7 copies per lane. Expected md5-identical.
- DIFFVAE_NA_HALO_PARTS=32 (timing knob): 512 B sticks, every reshape a view, more sticks through NP.

## Device runs (blx01)
- Overlay: /var/tmp/fasth3/t264/src/bb6b2c20e69 (models/ only) on C++ build t272/b @cae4b52657d (= t48 C++).
- Runner: /var/tmp/fasth3/t264/drv/run264.sh SRC 'arms' OUT 'profiled arms'; env drv/env.yaml.
- Job 022 (submitted 2026-10-08 11:35 UTC, -t 510): arms def vs h2d, both deep-profiled, seeds 0,1.
  Out /var/tmp/fasth3/t264/out_A (run.log, stage_tree_*.txt).
- Score: F=/var/tmp/fasth3; $F/t48/python_env/bin/python $F/t264/drv/cmp241.py $F/diffvae/ref $O/<arm> $O/cmp_<arm>.json 0,1
  and md5 lines in run.log (expect h2d md5 == def md5: exact change).

## Job 022 result (2026-10-08)
- def: decode 2.704/2.705 s (mean 2.705), md5 s0 2797bc15..., s1 13ee4b04...
- h2d: CRASHED at first decode: program.cpp:2471 'Statically allocated circular buffers on core range [0-0 - 3-0] grow to 4969472 B > 1572864 B L1'. The fused 2-D neighbor_pad sizes its CB from the 4 KB sticks (too big). Fix CB sizing (page-chunk the stick) before rerun. No drop.

## Next
- If h2d >= 5% faster (>= ~0.136 s): flip default (=0 off), ttp checks, land via -land branch + ttp push --detach.
- Else: try PARTS=32 arm (def vs h2d+parts32) as a separate job if the profile says NP is not stick-bound; else stop with notes.

## Drops
(none yet)
