# r02-b03-a01: destination-aware dual-NoC output drain: each output tile is written on the NoC with the shorter torus path to its DRAM bank (BRISC/NoC0 + idle NCRISC/NoC1)

## Motivation
On the round root (r01-b04-a04, 1.2132) the output drain is the kernel's tail: at h7168 W_DRAIN ends 1-4 µs after the
TRISC end (r01-b04-a03/a04 reflections: TRISC end 25.4-25.9, W_DRAIN end 26.9-29.6), and every 20-core node shows
the same ~210-260 GB/s aggregate write rate with a drain end that grows with core x/y (r01-b02-a02, r01-b03-a03).
That gradient is NoC0 link congestion, not DRAM banks (r01-b03-a03 de-phased banks and the drain barely moved).

Two earlier dual-NoC attempts, re-read with the BH geometry in hand:
- r01-b02-a04 (0.985, 20-core lineage) gave NCRISC the ODD 4-tile blocks. With num_tile_cols = 0 mod 8 and
  bank = page % 8, the odd blocks are exactly banks 4-7 (DRAM column x=9) and the even blocks banks 0-3 (x=0). So it
  sent every x=9 tile on NoC1 and every x=0 tile on NoC0. For the left-half workers (x=1..7) both are the long way
  round the torus (NoC1 to x=9 wraps west through x=0 and the whole right half; NoC0 to x=0 wraps east through the
  right half). The right-half workers got the short path for both. A link-load model (noc_drain_sim.py in this dir)
  gives the left cores ~2x the hops of the right cores, which matches the measured flip: left-half drain 28 µs vs 18.8 µs
  NoC0-only, right half equal or better. So that was a bad bank→NoC assignment, not proof that NoC1 is useless.
- r01-b03-a04 (80-core column split, +3%) used per-CB-position parity (≈ per-bank alternation, position-blind) and
  saw the gradient mirror instead of flatten.

Neither chose the NoC by destination.

## Mechanism
Per worker core and per DRAM bank, pick the NoC with fewer torus hops from this core to that bank's per-NoC
endpoint (NoC0: +x/+y, NoC1: -x/-y on the 17x12 BH grid; endpoints from blackhole_140_arch.yaml dram_views; physical
core coords from the NoC0 NOC_NODE_ID register, because my_x/my_y are translated and ignore column harvesting). The
result is an 8-bit mask computed on-device at kernel start (new header `kernels/dataflow/dit_rmsnorm_drain_noc_split.hpp`).
- worker writer (BRISC, NoC0): row-resident drain (output_cb already holds 2 full padded rows). Cumulative
  wait_front per block, write only tiles whose bank is NOT in the mask, flush, wait drain_sem, pop the row.
- reader (NCRISC, NoC1, idle once the input row is in): after its reads, cumulative wait_front per block, write
  tiles whose bank IS in the mask, write barrier, up drain_sem (local).
- factory: drain_sem on worker cores; reader gets output accessor CT args + output addr as common arg 6;
  enabled only for BH, DRAM-interleaved output, resident POST (no block_major_post / streaming), one row per worker
  (all four test shapes: 20 rows, 20 workers).
Model (noc_drain_sim.py, chip with workers at y=2/3): max per-link load 72 (NoC0 only) -> 31-35 (this), total hops
2236 -> 1506. The b02-a04 assignment had max load 68 and 2x per-core hops on the left half.

## Why this is not a repeat
r01-b02-a04 and r01-b03-a04 split by CB position (block or tile parity), not by destination. Here the split is
per (core, bank) by path length. Every core ends up with ~half its tiles on each NoC, each on the short side.
The drain_sem / no-pop-on-reader handshake is reused from r01-b02-a04 (it worked: no hang, correct data).

## Expected effect and risk
Drain tail shrinks; W_DRAIN max end - TRISC end should drop from 1-4 µs toward <1 µs on h6144/h7168, less on h3584.
Expect -0.5 to -2 µs per shape, score ~1.24-1.30. Risks: the hop model is wrong about the bottleneck (then neutral,
like b03-a04); NoC1 writes contend with something on NoC1 I haven't modelled. Accuracy can't change (same bytes).
Hang risk is low (protocol reused). Judge by per-core W_DRAIN/R_DRAIN end vs TRISC end, and whether the
x-gradient flattens.
