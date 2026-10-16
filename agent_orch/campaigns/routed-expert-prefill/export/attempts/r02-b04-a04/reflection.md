# r02-b04-a04 result: 1.5427 (ok): +1.3% over parent r02-b04-a03 (1.5234), tied with the campaign best r02-b03-a03 (1.5435, -0.05%)

## What happened vs expected
All 9 shapes are valid and **bit-identical to the parent** (PCC, max_abs and mean_abs equal to every
printed digit). That includes the four plain-SiLU shapes, so the op-local `re_act::silu_act` computes
exactly stock `calculate_silu<false>`. The ported binary bodies were already validated by r02-b03-a03. The
kernels JIT-compiled on the first run with no hang. (The eval rebuilt the host only because the shared eval
checkout had changed.)

Per shape vs the parent (µs, noise ±1%). Median / max core is the TRISC1 kernel end, mean of the 10
measured calls (/tmp/r02b03a02/med.py on reports/<node>/.logs/profile_log_device.csv):

| shape | activation | parent | this | Δ | median/max core parent -> this |
|---|---|---|---|---|---|
| gpt-oss | SwiGLU-OAI (ported) | 286.2 | 267.2 | **-6.6%** | 247/286 -> 245/267 |
| glm | SiLU (new) | 491.6 | 480.5 | **-2.3%** | 442/491 -> 443/480 |
| dsv3 | SiLU (new) | 564.6 | 558.5 | -1.1% | 499/564 -> 499/558 |
| dsv4-flash | clamped SiLU-GLU (ported) | 327.2 | 325.6 | -0.5% | 305/326 -> 304/325 |
| kimi-k3 | SiTU-GLU (ported) | 1365.0 | 1361.0 | -0.3% | 1319/1364 -> 1316/1360 |
| kimi-k2.7 | SiLU (new) | 773.1 | 771.4 | -0.2% | 605/772 -> 620/770 |
| minimax-m3 | SwiGLU-OAI (ported) | 383.2 | 382.4 | -0.2% | 351/383 -> 348/382 |
| mistral | SiLU (new) | 194.3 | 194.0 | -0.1% | **167/194 -> 161/193** |
| dsv4-pro | clamped SiLU-GLU (ported) | 1125.3 | 1128.5 | +0.3% | 1101/1124 -> 1092/1127 |

I expected ~1.54-1.55 and got 1.5427. That ties r02-b03-a03 by a different route: the b04 layouts
(1-row kimi-k3 -3%, mixed gpt-oss/dsv4-flash) in place of two-phase (kimi-k2.7 / dsv4-pro -4%).

## Why (best explanation, with evidence)
- **SiLU body: 34 -> 25 SFPU instructions per row.** I checked this with a standalone sfpi -O3 build of
  both functions (/tmp/dr_a04/cc/t.cpp, same compiler and -mcpu=tt-bh-tensix). Stock calculate_silu was
  ALREADY unrolled and replayed (a 32-instruction replay body + SFPSTORE/INCRWC per row). The saving
  is the 9 SFPLOADIs of 1/ln2, C0, C1 and C2 (two each) plus one SFPMOV. The rest is the same op sequence:
  SFPMAD(-x*1/ln2+127), the two SFPSWAP clamps, EXEXP/EXMAN/SHFT/CAST, a 2-MAD polynomial, SETEXP, ADD 1,
  ARECIP + 1 Newton step, MUL, STOCHRND. The body now fits the replay buffer in one 25-instruction record.
  So the SiLU pass is ~26% shorter per tile.
- **Where the SiLU saving shows:**
  - glm lost 11 µs of max (2 experts per group, compute-bound), but its median core did not move.
  - mistral's median core dropped 6 µs (-3.6%, one expert per group), but its max sits in the
    group-tail lottery (±5%), so the op time didn't move.
  - dsv3 / kimi-k2.7 are at the DRAM floor. The -1.1% on dsv3 is at the noise edge.
  - Expected size: ~18 SiLU tiles per core per expert x 32 rows x 9 instructions ≈ 5.2k cycles ≈ 4 µs
    per expert. That matches mistral's median shift (1 expert, plus the multiply pass) and glm's ~2 x 4-5 µs.
- **Binary port:**
  - gpt-oss's max fell 19 µs. Part of that is the parent's high draw: r02-b04-a01 had 277 with the same
    schedule, and the median moved only 2 µs.
  - minimax-m3 gained less than in r02-b03-a03 (-0.2% here vs -4.2% there). Its median moved 3 µs.
    r02-b03-a03's parent (393) was a high draw and this parent (383) a low one. Taken across nodes, the
    SwiGLU-OAI saving on minimax-m3 is about 1-2%, not 4%.
  - kimi-k3 / dsv4 are DRAM-bound and flat, as r02-b03-a03 found.

## Classification
win (+1.3% over the parent, bit-identical). It ties the campaign best but doesn't beat it outside noise.
The new SiLU piece is real but small (~4 µs per expert of SFPU time). It only shows on compute-bound
SiLU shapes (glm, and mistral's median core).

## What a child of this node should try next
1. **Combine with two-phase (r02-b03-a02's `plan_launch` / phase loop).** kimi-k2.7 (771 here vs 734 there)
   and dsv4-pro (1128 vs 1093) are the only shapes where this node is clearly behind the best. This node's
   compute kernel already holds every activation change, so porting the two-phase planner + phase-A
   semaphores onto this lineage should give ~1.555-1.56. Keep the 1-row layout as a phase-B/single-phase
   candidate only. Watch the planner prologue on kimi-k3 (r02-b03-a02 measured +20 µs median from the
   make_plan + plan_groups passes). Apply r02-b03-a02's next #1 (reject before modelling) at the same time.
2. **SiLU path, remaining SFPU cost:** the gate copy pass is still copy -> SFPU -> pack on MATH, then a
   separate FPU multiply_phase. r02-b01-a03's PACK-thread pipelining (SFPU on PACK, overlapped with the
   last gate/up K-block's matmuls) is orthogonal to the instruction cut. With re_act bodies on PACK, the
   replay buffer belongs to the PACK thread. Check that the packer does not use replay there
   (llk_pack.h does on some paths) before relying on it.
3. Don't spend more attempts shaving SFPU instructions without a precision change: SiLU (25),
   SwiGLU-OAI (31) and clamped (≈30) are now the bare op sequences. Only overlap (PACK thread) or fewer
   activation tiles can cut further.
4. The standalone sfpi build in /tmp/dr_a04/cc (t.cpp + inc.txt include list) compiles any SFPU body
   for BH in seconds. Use it to count instructions before an eval.
