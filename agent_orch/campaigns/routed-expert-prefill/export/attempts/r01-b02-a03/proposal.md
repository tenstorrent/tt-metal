# r01-b02-a03: port r01-b01-a02's row-shared x tilize onto this branch's collision-free row-group schedule

## Motivation
Both branches of the row-group idea are now near 1.45-1.47, and each carries a separate proven piece:
- r01-b02-a02 (parent, 1.4460): the b02 group-size rule + round-robin expert deal, a fixed staggered
  x sender, groups of >= 2 rows, and no bias reads for experts dealt to other groups. Against b01-a01
  it is ~3% faster on mistral (196 vs 200 us) and dsv4-flash (343 vs 353).
- r01-b01-a02 (1.4716): on the row-major path every gate/up K-block of x is read and tilized by ONE
  owner core per M-row (rotating column (x_blk + kb) % 11), which multicasts the bf8 tiles to the row.
  On top of b01-a01 it gained +2.4% overall: mistral +7.1%, gpt-oss +4.9%, dsv4-flash +3.5%, glm
  +2.2%, minimax +1.8%. It measured the per-block tilize + matmul/pack reconfig at ~5% of a
  compute-bound group pipeline (gpt-oss math-end times shifted 5% uniformly).

The parent still has every core in a row tilize the same x block (11x redundant) with a full
tilize init/uninit + matmul_block_init + pack reconfig in front of every gate/up K-block. Its
profile is in the same regime as b01-a01's: gpt-oss at 141 GB/s and the 4-expert shapes
latency/compute-bound.

## Mechanism
Apply b01-a02's change (`git diff n/r01-b01-a01 n/r01-b01-a02`) to the parent:
- compute `fused_swiglu.cpp`: `matmul_phase_fused_gu` tilizes only the blocks this core owns,
  one block ahead, `cb_x_rm -> cb_x_tl`; a global x_blk counter advances by g_num_blocks per chunk.
  Runtime arg 3 = this core's grid column (`my_nt_gu`).
- reader: the owner reads the RM block two blocks ahead into a single-block `cb_x_rm`. After its
  weight-sender work it multicasts `cb_x_tl` into every row core's `cb_in0_x` slot (loopback
  included) and then the valid sem. Row-mates ack the owner through the row NoC table. The TILE path
  keeps the fixed in0 sender byte-for-byte.
- factory: `cb_in0_x` is double-buffered on both layouts, `cb_x_rm` is a single block, and there
  is a new `CB_X_TL` (c_20, RM only). The footprint is updated, reader CT arg 36 = cb_x_tl, and the
  compute gets named `cb_x_tl`/`grid_x`.
- Merge points specific to b02: the compute keeps b02's `GroupDealer` (other-group experts get
  count 0, so they run no chunks and do not advance x_blk, the same in reader and compute). The owner
  index is `my_nt_gu`, which is gx in the factory and is the row NoC table's index.

## Why this is not a repeat
This combines two measured wins from different branches and has not been tried before. b01-a02 ran
on b01's schedule, where the x sender was fixed at column gy+1. The parent's schedule differs in its
group rule and deal, and in bias handling on gpt-oss. b02-a01's harmful *rotation* moved the x DRAM
read onto weight-sender cores in the critical path. Here the rotating owner's x read is issued two
blocks ahead and its multicast happens after the weight work, as in b01-a02, where it measured
positive.

## Expected effect and risk
- Expected: roughly b01-a02's per-shape gains over b01-a01 applied to the parent: mistral 196 ->
  ~183, gpt-oss 383 -> ~365, dsv4-flash 343 -> ~331, glm 504 -> ~493, neutral on the DRAM-bound
  shapes. That gives a score of ~1.48, which would be a new best.
- Numerics: bit-identical expected. PCC, max_abs and mean_abs should match the parent exactly.
- Risks: a merge mistake in the x_blk counting between the reader and compute would show as a hang
  or a PCC collapse. The L1 fit (+128 B per M*w tile) could move (M, w) on a shape, which would show
  as a changed accuracy digit.
