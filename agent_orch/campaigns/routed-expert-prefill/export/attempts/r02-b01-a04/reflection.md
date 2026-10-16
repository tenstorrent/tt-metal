# r02-b01-a04 result: 1.5657 (ok). New campaign best: +1.4% over r02-b03-a03 (1.5435), +2.5% over the parent r02-b01-a03 (1.5275)

## What happened vs expected
- All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to r02-b03-a03 and to the parent on
  every shape. So the op-local short SFPU bodies (sfpi-replayed SwiGLU-OAI / clamped SiLU-GLU, the
  65-instruction SiTU-GLU) give the same results when they run on the PACK thread inside the pipelined
  last gate/up K-block.
- No JIT error and no hang on the first device run. The PACK-thread replay recording did not disturb the
  fast-tilize pack replay, which is re-recorded at every owned x block.
- The 3-way merge was clean apart from fused_swiglu.cpp:
  - include block / PACK_ACT_* macros: now call `re_act::binary_act` / `binary_act_init` on PACK;
  - the deleted MATH `binary_activation_phase`: kept deleted.

Per shape (µs, chip mean; noise ±1%, ±5% on mistral / minimax-m3). The median/max core column is TRISC1
kernel end, mean of the 10 measured calls (/tmp/r02b03a02/med.py):

| shape | b01-a03 (parent) | b03-a03 (best) | this | vs b03-a03 | median/max core b03-a03 -> this |
|---|---|---|---|---|---|
| dsv4-pro | 1118.4 | 1093.2 | 1068.8 | **-2.2%** | 1086/1092 -> 1064/1068 |
| minimax-m3 | 372.0 | 376.9 | 368.3 | -2.3% | 346/376 -> 341/368 |
| gpt-oss | 330.8 | 267.6 | 262.2 | **-2.0%** | 244/267 -> 244/262 |
| kimi-k3 | 1353.5 | 1405.3 | 1386.0 | **-1.4%** | 1326/1405 -> 1310/1385 |
| dsv3 | 556.7 | 565.0 | 560.3 | -0.8% | 512/564 -> 508/560 |
| dsv4-flash | 326.3 | 326.2 | 325.7 | 0 | 310/325 -> 303/325 |
| glm | 475.4 | 492.2 | 492.2 | 0 | 445/492 -> 440/492 |
| kimi-k2.7 | 755.8 | 734.2 | 739.3 | +0.7% | 731/733 -> 736/738 |
| mistral | 184.5 | 197.9 | 189.1 | -4.5% (lottery range) | 169/197 -> 165/188 |

I expected ~1.55-1.56 and got 1.5657. Each piece kept its gains:
- the schedule: gpt-oss 262 (not the parent's 331); kimi-k2.7 and dsv4-pro tail-free;
- the pipeline: dsv4-pro, minimax-m3, gpt-oss and kimi-k3 are 1.4-2.3% under b03-a03.

## Why (profiler evidence)
MATH zones (/tmp/r02b01a02/zones.py on reports/r02-b01-a04/.logs/profile_log_device.csv):
- **The activation is gone from the critical path on every binary shape.** RE-ACT is 0.0-0.1 µs on
  kimi-k3, dsv4-flash/pro, mm3 and gpt-oss. The pipelined SFPU now hides behind the last-block FPU
  work.
- **The short bodies also shrink the exposed tail.** The last subblock's SFPU, which b01-a03 found
  leaking into DN, is about 30% cheaper with the short bodies. That explains why dsv4-pro gained more
  here (-2.2%) than either piece alone (b01-a03 -0.9%, b03-a03 +0.9%).
- **kimi-k2.7 is flat to slightly slower.** It is tail-free and DRAM-bound at ~455 GB/s, so a compute
  saving cannot move it. glm and dsv4-flash are the same.
- **New measured cost: the planner prologue.** first-GU-start (kernel start -> first RE-GU) is:

  | | kimi-k3 | dsv3 / glm / k2.7 / dsv4 | 4-expert shapes |
  |---|---|---|---|
  | parent (old picker) | 6.4 µs | 3.2-3.9 µs | 2.5 µs |
  | this node (b03 plan_launch) | **29.2 µs** | 10-11 µs | 5-6 µs |

  - That is ~23 µs on kimi-k3 (1.7% of its time) and ~7 µs on dsv3 / glm / dsv4-flash (1.3-2%).
  - Every core pays it before its first DRAM read, so it is critical-path time.
  - This confirms r02-b03-a02's suspicion with a direct measurement.

## Classification
win (+1.4% geomean over the previous best, bit-identical numerics). It combines the schedule line
(mixed layouts + two-phase) with the compute line (short SFPU bodies + PACK-thread pipeline), and the
gains add up.

## What a child of this node should try next
1. **Cut the planner prologue (measured 5-29 µs per call, ~+0.7-1% geomean, kernel-only).**
   - In `adaptive_chunk::make_plan`, reject two-phase before any `model_phase` when the leftover can't
     fit one phase-B round (r02-b03-a02 next #1).
   - Cache `tiles_of(e)` once into a local `uint8_t tiles[E]` array. Every planner pass re-reads the
     counts L1 through `expert_count_tiles`.
   - Have `plan_launch` reuse the uniform pick instead of rescheduling the 3 uniform layouts twice.
   - Better still: compute the plan on ONE RISC (e.g. the counts-reader core's NCRISC) and add it to the
     counts multicast as a small extra page, so no core models anything. Compute UNPACK then reads it
     from L1.
   - Judge by first-GU-start in the zones: target back to ~3-6 µs. kimi-k3 should drop ~20 µs.
2. **SiLU shapes (dsv3, kimi-k2.7, glm, mistral)** still run stock silu in the pipelined last block. Copy
   the silu body into binary_act_sfpu.h-style op-local code and hoist its constants, as b03-a03 did for
   the binary ops. Expect little: dsv3 / glm / k2.7 are DRAM- or tail-bound. mistral is the one
   compute-bound SiLU shape.
3. The remaining big waste is the max − mean group tail on dsv3 / glm (MEAN core end 436 / 382 vs
   CRIT 559 / 491, whole-expert imbalance 2/2/2/1/1). An N-split of a last-wave expert across two groups
   (r02-b02-a03 next #2) is still the only untried lever for it.
4. Don't add more to the activation: RE-ACT is ~0 on every binary shape now.
