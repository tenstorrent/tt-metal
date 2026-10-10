# #374 conv3d vol2col input reuse: design (step 1, CPU only)

Base: ttp/t48-ltx25-integrated 90ed8257bac (worktree head f6547442b30).
Head blockings (models/tt_dit/utils/conv3d.py, since dad2d7a90cc), tuple = C_in_blk, C_out_blk, T, H, W:

| layer | input T×H×W, C | blocking | measured (job 452 era, clamp) |
|---|---|---|---|
| s2_res (res512b) | 75×34×30, 512→512 | 64,256,2,4,4 | 16319 µs |
| s3_res (res256)  | 147×34×30, 256→256 | 64,256,2,4,4 | 6372 µs |
| s4_res (res128)  | 147×68×60, 128→128 | 128,64,6,4,8 | 8170 µs |

All 3x3x3, stride 1, dilation 1, L1 prefetch on.

## How the reader moves bytes today (reader_vol2col.cpp)

Loop: batch → c_in_blk → c_out_blk → t_blk → h_blk → w_blk. Per output block:
1. DRAM→L1 gather into a shard of T_shard×H_shard×W_shard sticks (T_shard = T_blk+2 etc.).
   The first w_blk of an h_blk gathers the full shard; later w_blks shift the kW−1 = 2 retained
   W columns left inside L1 (`shift_retained_w_columns`) and gather only W_blk new columns.
   H rows are gathered incrementally per output h, so the gather overlaps vol2col.
2. vol2col: per patch, kT·kH reads of kW·C_in_blk bytes, L1→CB (`ChunkWriter`, 32-patch chunks).

Nothing is reused across h_blks: each h_blk re-gathers the kH−1 = 2 halo rows that the
previous h_blk already had, and every w_blk re-gathers its own T×H halo.

### res256 / res512b (same 34×30 plane, C_in_blk 64 → 128 B sticks)

Per t_blk, per (c_in_blk, c_out_blk) pair, per core: 9 h_blks × 8 w_blks.
- Gather: per h_blk 4×6×6 = 144 sticks (first w_blk) + 7×(4×6×4) = 672 → 816 sticks;
  last h_blk is partial (34 = 8·4+2 → 4×4×… = 544). Total ≈ 7072 sticks ≈ 905 KB,
  plus ≈ 1512 L1 shift commands (256 B each).
- vol2col: 288 reads × 384 B per 32-patch block ≈ 110.6 KB; ≈ 20.7k NOC commands per t_blk.
- DRAM sticks per output voxel ≈ 3.6 against a unique footprint of 1.5 (T 4/2 × halo).
- res512b repeats the whole gather for each of its 2 c_out blocks (the shard is rebuilt per
  c_out_blk), so its gather bytes per t_blk are 2× res256's per c_in_blk.

### res128 (68×60 plane, C_in_blk 128 → 256 B sticks)

T_shard 8, H_shard 6, W_shard 10; 17 h_blks × 8 w_blks (60 = 7·8+4).
- Gather per h_blk: 8×6×10 + 7×(8×6×8) = 480 + 2688 = 3168 sticks; 17 h_blks → 53.9k sticks
  ≈ 13.8 MB per t_blk per c_out_blk (2 c_out blks).
- vol2col per 192-patch block: 192·9 = 1728 reads × 768 B.

## Option A (chosen): full-width H-row ring ("row ring")

Shard becomes [T_shard][H_ring = H_shard][W_full], W_full = (W_out_core−1)·s + kW
(= the core's whole padded W extent; 32 for res256/res512b).

- At each new t_blk (and c_in/c_out blk) the ring is empty.
- Each h_blk gathers only the input rows it does not have yet (H_shard on the first h_blk,
  then H_blk new rows), full width, into slot = (row − first_row) mod H_ring. A run that
  wraps is split in two `gather_rows_to_shard` calls (h_shard_start' = row − slot).
- w_blks gather nothing and shift nothing: vol2col reads straight from the ring with
  w_base = (w − w_out_core_start)·s and a per-kh slot map.
- The patches, their order and the bytes in each patch are unchanged, so the CB content
  is identical and compute/writer/reducer are untouched: the output is bit-identical.
- Overwrite safety: the rows a new h_blk overwrites were last read by the previous h_blk's
  vol2col, whose reads finish before ChunkWriter pushes its last chunk (it waits on reads
  before cb_push_back), and the next h_blk's gather starts after that.

Effect on res256 per t_blk: DRAM sticks 7072 → 4×(6+8·4)×32 ≈ 4864 (−31%), with the last
h_blk partial ≈ 4608 (−35%); 1512 shift commands → 0. Total reader NOC commands −13.5%.
#363 (T1H8W4 → T2H4W4) cut gather sticks ~21% and shifts ~18% for a 15% layer gain, with
vol2col unchanged; by that ratio row ring should give ~20% on res256. This is an estimate:
compute (fused tilize+matmul, ideal ≈ 13.8k of ≈ 26.5k cycles per block at 900 MHz) may
bound part of it.

L1: res256 ring 4×6×32×128 B = 98 KB vs 18 KB today (+80 KB). The factory checks it against
`l1_prefetch_max_bytes` and falls back to today's path if it does not fit.
res512b: same ring size, fits the same way. res128: 8×6×62×256 B = 761 KB, does not fit next
to its 192-patch CBs; it needs a W-tiled ring (ring width = a few w_blks) and is out of
scope until res256 shows ≥15%.

Opt-in: env `TT_CONV3D_ROW_RING=1`, read in the program factory, passed as reader
compile-time arg 47 (TensorAccessorArgs follow it). Only for stride 1, dilation 1, L1
prefetch on, non-coalesced gather path. The program cache key does not include the env,
so A/B arms run in separate processes.

## Option B (considered, deferred): rolling T window

Keep kT−1 = 2 frames from the previous t_blk and gather only T_blk new frames. At the current
core split (c_in_par 4, t_par 30 → 3 t_blks per core for res256) the ring must hold a whole
H×W plane per frame, and making t the inner loop loses the W sliding: ≈ 3.0 sticks per voxel,
no better than today. It pays only with cores split over H/W and a long T range per core
(≈ 2.27 sticks/voxel), which would need a new core split and is not bit-risk free in the
reducer. Deferred.

## Not redone

#98 (fused neighbor_pad + conv3d) and #335 (fidelity, rejected) are separate levers.

## Plan

1. Prototype row ring on res256 (opt-in), unit test bit-exact vs flag off on
   create_submesh(2,4) of the full mesh, then a single-layer A/B job on blx01.
2. Only if res256 gains ≥15%: res512b (same code path) and a W-tiled ring for res128,
   then a module A/B.
