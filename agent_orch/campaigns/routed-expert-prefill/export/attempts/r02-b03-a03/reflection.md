# r02-b03-a03 result: 1.5435 (ok), new campaign best (parent r02-b03-a02 1.5278, +1.0%; previous best r02-b04-a01 1.5314)

## What happened vs expected
All 9 shapes are valid and **bit-identical to the parent** (PCC, max_abs and mean_abs equal to every
printed digit on every shape, including the five binary-activation shapes). So the op-local
SwiGLU-OAI / SiTU-GLU / clamped SiLU-GLU compute exactly the stock math. That covers the hoisted
constants, the Prgm1/2 CRegs, the SFPSWAP clamps and the sfpi-replayed unrolled bodies. The kernels
JIT-compiled first try. There was no hang, so the replay-buffer overwrite of the MATH FPU region is
harmless: matmul_block_init re-records it before every matmul phase.

Per shape vs the parent (µs, noise ±1%). Median / max core is the TRISC1 kernel end, mean of the 10
measured calls (/tmp/r02b03a02/med.py on reports/<node>/.logs/profile_log_device.csv):

| shape | activation | parent µs | this µs | Δ | median/max core parent -> this |
|---|---|---|---|---|---|
| gpt-oss | SwiGLU-OAI (+bias) | 281.1 | 267.6 | **-4.8%** | 250/281 -> 244/267 |
| minimax-m3 | SwiGLU-OAI | 393.4 | 376.9 | **-4.2%** | 355/393 -> 346/376 |
| dsv4-flash | clamped SiLU-GLU | 330.5 | 326.2 | -1.3% | 313/330 -> 310/325 |
| dsv4-pro | clamped SiLU-GLU | 1083.5 | 1093.2 | +0.9% (noise) | 1078/1082 -> 1086/1092 |
| kimi-k3 | SiTU-GLU | 1404.5 | 1405.3 | 0 | 1327/1404 -> 1326/1405 |
| dsv3 / kimi-k2.7 / glm / mistral | SiLU (code unchanged) | | | +1.6 / -0.2 / -1.6 / +0.6% | run-to-run noise |

Expected: gpt-oss -5..-7% (got -4.8%), minimax-m3 -2..-4% (got -4.2%), dsv4 -1..-2% (got ~-1/+1,
noise), kimi-k3 up to -2..-3% (got 0).

## Why (best explanation, with evidence)
- **SwiGLU-OAI shapes gained the predicted amount.** Per row, the SFPU loop went from 54 RISC-issued
  instructions (18 SFPLOADI) to a 31-instruction body that sfpi records once per face and replays,
  plus ~4 RISC instructions per row. Measured stock cost was ~60 cycles/row. If the activation is
  ~55 µs per expert on gpt-oss (r02-b04-a02 zones), a 25-35% cut is ~14-19 µs. gpt-oss's max core
  dropped 14 µs and its median 6 µs. minimax-m3 (39 µs per expert) dropped ~9 µs (median) to 17 µs
  (max). Both are compute-bound with one expert per group, so the saving goes 1:1 to the critical
  path.
- **kimi-k3 did not move (median 1327 -> 1326).** SiTU went 77 -> 65 instructions per row (-16%,
  no replay because the body is longer than 32). Two non-exclusive reasons:
  - (a) SiTU may be stall-bound rather than issue-bound: its dependent mad chains (two 6-deep
    tanh Horner chains, two sfpnops in the body) leave issue slots that the removed SFPLOADIs were
    filling for free.
  - (b) kimi-k3 sits at ~400 GB/s with 5 desynchronised groups, so one group's shorter activation
    mostly hands DRAM time to the other groups.
  The zones would separate (a) from (b); this node has none.
- **Clamped SiLU-GLU (dsv4)** went 44 -> 35 per row, replayed. The activation is only ~15 µs per
  expert there and both shapes are near DRAM-bound, so ≤ 1% was the most it could show. It sits in
  noise.
- SiLU shapes run identical code. Their ±1.6% is the known tail lottery (median core: dsv3 504 ->
  512, glm 451 -> 445).

## Classification
win: +1.0% geomean over the parent and the new campaign best (1.5435). Two shapes moved well
outside noise (gpt-oss -4.8%, minimax-m3 -4.2%), with bit-identical numerics and no schedule
change.

## What a child of this node should try next
1. **The SiLU shapes' activation is untouched.** silu_tile runs in the gate copy pass of
   matmul_phase_fused_gu (stock ckernel silu, MATH thread), and multiply_phase is an FPU mul. Do
   the same treatment there: copy the silu SFPU body into the op directory, hoist its constants
   and unroll it for replay. First measure it: put a zone around the gate copy loop, since it was
   counted inside RE-GU so far. This touches dsv3, kimi-k2.7, glm and mistral, which nothing in
   this node affected.
2. **SiTU-GLU (kimi-k3):** check whether it is issue-bound before trimming more. Re-add the
   r02-b04-a02 RE-ACT zone (MATH thread) and compare the per-expert activation time against ~40 µs.
   If it did not shrink, the body is stall-bound. Then interleave two rows per iteration (the
   _sfpu_tanh_polynomial_x2_ pattern in ckernel_sfpu_tanh.h) so one row's mads fill the other's
   stall slots. If it did shrink, kimi-k3 is DRAM-bound and further SiTU work is pointless.
3. **gpt-oss is still mostly activation + per-expert fixed cost.** The SwiGLU body is now 31 SFPU
   instructions. Further cuts need a different exp/recip formulation, which the precision rule
   forbids. The remaining lever is overlap: run the SFPU part from the PACK thread so MATH's
   copy_tile / add_bcast of the next chunk overlaps it. That is small, about 2 copies per tile.
4. The parent's planner-prologue trim (r02-b03-a02 next #1: reject in make_plan before modelling)
   is still untried and orthogonal to this node.
5. binary_act_sfpu.h is self-contained. A node on another branch can copy it, plus the two
   BINARY_ACT_* macros and the include in fused_swiglu.cpp, without merging schedules.
