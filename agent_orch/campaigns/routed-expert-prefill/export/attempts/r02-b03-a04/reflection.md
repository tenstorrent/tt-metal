# r02-b03-a04 result: 1.5630 (ok), new campaign best (parent r02-b03-a03 1.5435, +1.3%)

## What happened vs expected
All 9 shapes are valid and **bit-identical to the parent** (PCC, max_abs and mean_abs equal to every printed
digit). So these all reproduce the parent's numerics exactly:
- the op-local SFPU bodies run from the PACK thread (sfpi replay on TRISC2, Prgm1/2 set from PACK),
- the per-subblock partials hand-off in the last gate/up block,
- the bias add moved into the pipeline (gpt-oss),
- SiLU via silu_tile_pack.

The first device run had no JIT error and no hang. A local JIT harness (/tmp/r02b03a04_k, all 4 variants
x 3 TRISCs, RM and TILE x) caught nothing, so the device run was the only run.

Per shape vs the parent (µs; noise ±1%, ±5% on the one-expert-per-group shapes). "median/max core" is the
TRISC1 kernel end, mean of the 10 measured calls (/tmp/r02b03a02/med.py <node>):

| shape | parent µs | this µs | Δ | median/max core parent -> this |
|---|---|---|---|---|
| mistral (SiLU) | 197.9 | 189.5 | -4.3% (partly lottery) | 169/197 -> 162/189 |
| dsv3 (SiLU) | 565.0 | 549.2 | -2.8% | 512/564 -> 499/548 |
| gpt-oss (SwiGLU-OAI + bias) | 267.6 | 261.9 | **-2.1%** | 244/267 -> **233**/262 |
| minimax-m3 (SwiGLU-OAI) | 376.9 | 369.4 | **-2.0%** | 346/376 -> **336**/369 |
| dsv4-pro (clamped SiLU-GLU) | 1093.2 | 1077.5 | -1.4% | 1086/1092 -> 1072/1076 |
| kimi-k3 (SiTU-GLU) | 1405.3 | 1399.5 | -0.4% | 1326/1405 -> 1321/1399 |
| dsv4-flash (clamped) | 326.2 | 326.1 | 0 | 310/325 -> 304/325 |
| glm (SiLU) | 492.2 | 497.2 | +1.0% (lottery: median 445 -> 439) | 445/492 -> 439/497 |
| kimi-k2.7 (SiLU) | 734.2 | 740.9 | +0.9% | 731/733 -> 738/740 |

Expected +1..+2% geomean, about 1.56-1.57. Got +1.3% (1.5630). The **median core moved down on 8 of 9 shapes**
(gpt-oss -4.5%, mm3 -2.9%, dsv3 -2.5%, mistral -4%, dsv4-pro -1.3%, dsv4-flash -2%), so the gain is a real
shift of the work rate, not the tail lottery. kimi-k2.7 is the exception (+1%). It is two-phase and DRAM-bound,
with all rows ending within 3 µs, so its +7 µs is most likely run-to-run DRAM noise.

## Why (profiler evidence)
MATH zones (/tmp/r02b01a02/zones.py on reports/r02-b03-a04/.logs/profile_log_device.csv). RE-GU now
includes the pipelined activation. RE-ACT is only the SiLU multiply pass (binary shapes: 0).

| shape | mean core: GU / ACT / DN (µs per core per call) | critical core end |
|---|---|---|
| gpt-oss | 154 / 0 / 74 | 261 |
| minimax-m3 | 228 / 0 / 110 | 368 |
| kimi-k3 | 732 / 0 / 525 | 1398 |
| dsv4-pro | 694 / 0 / 357 | 1076 |
| dsv3 | 284 / 8.4 / 126 | 548 |

- **Binary shapes:** the parent ran the (already ~30% shorter) SFPU body serially after the K-loop. Now each
  subblock's SFPU runs on PACK under the next subblock's last-block FPU work.
  - gpt-oss (pcM 3-4, 9-wide N, 12+ last-block subblocks) and mm3 hide the most: -11 / -10 µs median core.
  - The overlap and the shorter bodies add up, as predicted. The overlap is bounded by last-block FPU time
    per subblock, not by SFPU length.
  - kimi-k3 / dsv4-flash: the median moved 5-6 µs, but the max did not. These are near the DRAM ceiling
    (~400 GB/s), so time saved on one group goes to the other groups' streams (same pattern as r02-b01-a03).
- **SiLU shapes:** silu_tile left MATH's copy pass and runs on PACK, overlapped with the last block.
  - dsv3's median core fell 13 µs and mistral's 7 µs.
  - glm's median fell 6 µs but its max rose (the lottery).
- DN is about one third of the per-core time on every shape. That matches its MAC share, so there is no
  sign of the down-ring "head start" problem r02-b01-a03 saw, beyond noise.

## Classification
win: +1.3% geomean and a new campaign best (1.5630), bit-identical numerics. It combines two earlier wins on
different branches: r02-b01-a03's pipelining and r02-b03-a03's SFPU bodies. Their effects add up.

## What a child of this node should try next
1. **SiLU body treatment (still untried).** silu_tile_pack runs the stock calculate_silu (exp + recip
   per row, constants reloaded per row). Copy it into binary_act_sfpu.h, hoist constants, and unroll for
   replay, as r02-b03-a03 did for the binary ops. It now runs on PACK, so every cycle saved on the last
   subblock (the one that can't hide) comes straight off the GU→DN critical path on dsv3 / glm /
   mistral / kimi-k2.7. Expected small: 0.5-1%.
2. **The last subblock's activation is still exposed** (~3 tiles per tile-row at the end of GU). Order the
   last block so the final LW-subblock is narrow (e.g. LW-tail = 1-2 tiles) or start the down matmul's first
   K-block while PACK finishes it. That is worth ~2-5 µs per expert on gpt-oss / mm3 / kimi-k3.
3. **Planner prologue trim** (r02-b03-a02 next #1) is still untried. kimi-k3's first GU starts 29 µs after
   kernel start, against ~10 µs on the other shapes. That is the 28-expert planning (make_plan + plan_groups +
   tiles_of walks) on every RISC. Cache tiles in a local array and reject two-phase before any modelling. For
   kimi-k3, +1% is plausible, but only if its median moves: the shape is DRAM-bound.
4. Don't retry: DM look-ahead into the activation gap (r02-b02-a03), the writer-owned down stream
   (r02-b04-a02), request VCs (r02-b02-a02).
5. Code notes: binary_act_sfpu.h is now compiled on TRISC_PACK only. PACK_ACT_INIT / PACK_ACT_TILE in
   fused_swiglu.cpp call re_act::binary_act_init / binary_act. MATH runs no SFPU op. The local compile
   harness is /tmp/r02b03a04_k (kc.sh / run.sh, which takes CT args; fs.cpp is the kernel copy it compiles).
