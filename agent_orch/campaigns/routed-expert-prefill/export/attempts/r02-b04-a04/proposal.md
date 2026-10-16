# r02-b04-a04: same-math SFPU activations in fewer instructions for every activation: port r02-b03-a03's op-local binary bodies (SwiGLU-OAI / SiTU-GLU / clamped SiLU-GLU) and extend the treatment to the plain-SiLU gate pass (constants in LRegs / Prgm1-2 CRegs, unrolled for sfpi replay)

## Motivation
- The activation is serial MATH-thread SFPU work between the gate/up K-loop and the down phase
  (r02-b01-a02 / r02-b04-a02 zones: ~55 µs per expert on gpt-oss, ~39 on minimax-m3, ~40 on kimi-k3,
  ~15-23 on dsv4). The stock SFPU bodies re-load every fp32 constant (two SFPLOADIs each) on every row
  and are instruction-issue bound (~1 cycle per instruction).
- r02-b03-a03 (campaign best, 1.5435) rewrote the three binary bodies with the same math in fewer
  instructions and measured gpt-oss -4.8% and minimax-m3 -4.2%, bit-identical. That header is
  self-contained, and this branch (r02-b04-a01 layouts + 1-row layout) does not have it.
- **The plain-SiLU path has never been touched.** dsv3, kimi-k2.7, glm and mistral run stock
  `silu_tile` on the MATH thread in the gate copy pass after the gate/up K-loop (pcM x pcN_gu tiles
  per core per expert, e.g. 18 on mistral/glm). Stock `calculate_silu<false>` is x * _sfpu_sigmoid_<false>(x)
  = x * recip_iter<1>(1 + exp_21f(-x)): the same exp_21f + Newton reciprocal that the clamped
  SiLU-GLU body uses, which r02-b03-a03 cut from 44 to 35 instructions per row and made replayable.
  mistral (1 expert per 2-row group) and glm (2 per group) are compute-bound (~50% / ~25% above
  their DRAM floor), so the SiLU pass is on their critical path.

## Mechanism
Compute kernel only (JIT, no host rebuild):
1. Port `device/kernels/compute/binary_act_sfpu.h` and the `BINARY_ACT_*` macros / include from
   r02-b03-a03 verbatim (the activation code of `fused_swiglu.cpp` is identical on both lineages).
2. Add `re_act::silu_act<fp32, 8>` to that header: bf16-dest body =
   `x * sfpu_reciprocal_iter<1>(1.0f + exp_21f(-x, Prgm1=1/ln2, c0, c1, Prgm2=C2))`, bf16 RN convert,
   with c0/c1 hoisted into LRegs once per face and the loop unrolled (`#pragma GCC unroll 8`) so sfpi
   can replay it. Same operations, same order, same fp32 values as stock calculate_silu (the
   exp_21f helper is the one r02-b03-a03 already validated bit-identical inside clamped SiLU-GLU).
   The fp32-dest path calls stock calculate_silu.
3. `fused_swiglu.cpp`: the gate copy pass calls `re_act::silu_act` through SFPU_UNARY_CALL instead of
   `silu_tile`; after `silu_tile_init()` (Prgm0 = 2.0 for the reciprocal) MATH sets Prgm1/Prgm2.
   Nothing else in the kernel programs those CRegs. The replay region it overwrites is the MATH
   matmul one, which matmul_block_init re-records before every matmul phase; the multiply phase
   (eltwise binary) and copy_tile use no replay.

## Why this is not a repeat
- The binary part is a port of r02-b03-a03 (measured win) onto the b04 layout lineage, which has
  never had it. The SiLU part is new: no node has changed the SiLU implementation
  (r02-b03-a03's reflection lists it as next #1; r02-b01-a03 only moved stock silu to PACK).
- Orthogonal to this branch's schedule work (mixed-height / 1-row layouts): the planner and DM
  kernels are untouched.

## Expected effect and risk
- gpt-oss ~-5%, minimax-m3 ~-4% (as in r02-b03-a03). SiLU shapes: mistral -2..-4%, glm -1..-3%,
  dsv3 / kimi-k2.7 ~0..-1% (near the DRAM floor). kimi-k3 / dsv4 ~0. Score ~1.54-1.55.
- Accuracy must be bit-identical to the parent on every shape (PCC / max_abs / mean_abs to every
  printed digit). Any digit change on a SiLU shape means the SiLU body differs from stock.
- Risks: JIT compile error (sfpi register allocation), replay interaction (a hang or garbage would
  point at it).
