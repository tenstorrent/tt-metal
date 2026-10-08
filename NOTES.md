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

## Verdict (2026-10-08, run 1095): STOP, every in-scope lever is under the 5% bar (0.135 s)
Nothing landed. bb6b2c20e69 (DIFFVAE_NA_HALO_2D, opt-in) stays on this branch only: it CRASHES when enabled (L1 CB).

Root cause of the crash (code read, neighbor_pad_async_program_factory.cpp ~L326-338): in the fused 2-D mode
the H-fabric cores keep a recv CB that holds ALL corner sticks of all their outer dims (no reuse, because
fabric can deliver outer dim N+1 before the reader drains N). Bytes = outer_per_core * pad_h * corner bytes,
independent of the stick split, so "chunk the stick" does not fix it. Only 4 H-fabric cores (2 links); more
links are capped by the 13-wide grid (10 H cores max -> still ~2 MB > 1.5 MB). A fix needs a flow-controlled
recv protocol in the kernels plus a host rebuild.

Stage-5 geometry (run.log): volume (145,272,480), H x4, W x8, brick (2,4,4), window 11 -> halo 2 bricks
on H and W. Owned bricks 73x17x15 = 18615 per chip; exchanged 73x21x19 = 29127 (1.56x); phased 26280 (1.41x).

Band-copy accounting per lane (unit = one owned band; measured ~2.3-2.4 ms/unit from the deep profile:
untilize 1.8 ms/1.0, gather 3.0 ms/1.41, unattributed 11.6 ms/(2 x (1+1.56))):
- today: untilize 1, rows->grid5 1, grid5->split 1, NP-H 1.24, NP-W 1.56, site_major 1.56, gather 1.41 = ~8.8 units (~21 ms/lane, 42 ms/block)
- h2d fixed (C++): saves grid5 copy (1) + one NP pass (~1.24) = ~2.2 units = ~11 ms/block = ~0.09 s (3.3%). Under the bar, needs a rebuild: not done.
- direct rows->split only (Python, exact): saves 1 unit/lane = ~5 ms/block = ~0.04 s (1.5%).
- PARTS=32 (sticks of C = 256 B, every reshape a view): saves ~3.6 units/lane but NP's local copy moves one stick
  per read barrier with a 2-page CB, 16x more sticks (~5.4k/core/NP at ~1 us) -> +~8 ms/lane. Wash. Not run.
- edge-only exchange + one concatenated gather table (Python, exact): slices/concats/NPs on the H edge slab
  (2x2/17 = 24% of band) and the W slab with corners (21x4/255 = 33%), then a 1.56-unit table concat:
  ~6.8 units -> saves ~2 units/lane = ~0.075 s (2.8%) before the extra ~12 small-op dispatches. Under the bar.

What would clear the bar (all need new device code, beyond this task):
- multi-table gather (embedding that reads [band | halo_top | halo_bot | halo_left | halo_right] by index) on top of
  the edge-only exchange: drops the 1.56-unit table concat -> ~5.2 units, ~17 ms/block, ~0.14 s (5.1%). Marginal.
- NA reader taking phased K/V sites straight from the tiled band + halo buffers (no untilize/reshape/gather):
  ~42 -> <10 ms/block, ~0.25-0.3 s (9-11%). Large kernel project (site-level phase offset inside the reader).

## Drops
None. Job 022 (blx01, 2026-10-08 11:35 UTC) ran to the end; the h2d arm failed on L1 allocation, not a drop.

## Cleanup
blx01 /var/tmp/fasth3/t264 (overlay src 132 MB, out_A 871 MB incl. yuv decodes, drv) removed 2026-10-08.
Evidence kept in tt-project/t264/ (stage_tree_def.txt, run.log, decode times, driver).
