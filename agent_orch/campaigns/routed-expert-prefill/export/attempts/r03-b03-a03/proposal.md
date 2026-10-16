# r03-b03-a03: plain SiLU runs as a fused binary SiLU-GLU SFPU op on PACK (silu(gate)*up in one pass, pipelined into the last gate/up K-block), dropping the separate FPU multiply pass; code base = best node r03-b02-a02

## Motivation
- The parent (r03-b03-a02, 1.4983) regressed because its reader read gate+up on one NoC. Its own reflection
  says to start from better code. I take the code of the campaign best, r03-b02-a02 (1.5952): r03-b02-a01's
  planner + PACK-thread pipelined activation + x owners off weight senders. I add one mechanism on top.
- On the four SiLU shapes (dsv3, glm, kimi-k2.7, mistral) the activation is still two passes. From
  r03-b02-a02's MATH zones (reports/r03-b02-a02, /tmp/r03b03a03_tl.py), per expert:
  - silu(gate) runs on the PACK thread, pipelined into the last gate/up K-block. It is packed to bf8
    `cb_gate_intermed`.
  - Then **RE-ACT is ~5 µs per expert**: `multiply_phase`, a serial FPU pass (unpack bf8 silu + bf16 up
    partials, LoFi mul, pack bf8 `cb_activated`). The down phase cannot start until it ends.
  - dsv3/glm: 2 experts per row → ~10 µs of ~497 µs per core. kimi-k2.7: 3 experts → ~15 µs. mistral:
    one expert, ~5 µs of 165 µs.
  - The binary-activation shapes (SwiGLU-OAI, SiTU-GLU, clamped SiLU-GLU) already compute gate×up inside
    their PACK-thread SFPU op. Their RE-ACT is ~0 (r03-b02-a01 reflection).
- A clamped SiLU-GLU body already exists op-locally (binary_act_sfpu.h). Plain SiLU-GLU is the same body
  without the two clamps.

## Mechanism
Kernel-only (JIT), compute kernel only:
1. `kernels/compute/binary_act_sfpu.h`: new `SILU_GLU` variant, `silu_glu_bf16`.
   - silu(gate) = gate × reciprocal_iter<1>(1 + exp_21f(−gate)), rounded to bf16. This is step for step
     the stock `calculate_silu<false>`; the same op-local exp_21f the clamped body uses.
   - Then × up in fp32, rounded to bf16, stored to the out dst slot. The packer packs it to bf8
     `cb_activated` as before.
   - Init: `sigmoid_init<false>` plus Prgm1/Prgm2 constants, as in the clamped variant.
2. `fused_swiglu.cpp`: when no variant define is set (plain SiLU), FP32_DEST_ACC_EN == 0, no FUSE_BIAS and
   ARCH_BLACKHOLE, define `SILU_GLU`, which turns on `FUSED_BINARY_ACT`.
   - The existing binary path then runs it: PACK-thread SFPU per LW subblock of the last gate/up K-block,
     straight into `cb_activated`, with partials_up popped there.
   - `multiply_phase` is skipped (RE-ACT ≈ 0).
   - fp32 dest, other archs and anything else keep the old silu + multiply_phase path unchanged.
   - SiLU has no bias variant (validation).

Precision: never lower.
- silu(gate) is bit-identical to before up to its bf16 rounding.
- Before, it was then rounded to bf8 and multiplied by up in a LoFi FPU multiply (reduced mantissa).
- Now the multiply is fp32 on the SFPU, followed by the same bf16 → bf8 output pack. The activated
  intermediate keeps its format (bf8 `cb_activated`).
- Fidelity of every matmul, accumulation formats and approximation modes are unchanged. Only the
  elementwise product gets more precise. That is the same structure the other three activations use.
- Numerics will differ slightly from the baseline (not bit-identical). PCC and mean_abs should be equal or
  better.

## Why this is not a repeat
- r02-b04-a04 shortened the SiLU SFPU body (unary silu, still followed by multiply_phase).
- r02-b01-a03 / r02-b03-a04 / r03-b02-a01 pipelined the activation onto PACK. For SiLU they kept
  silu → bf8 → FPU multiply_phase, so the SiLU shapes still pay RE-ACT (~5 µs per expert, measured above).
- Nobody has turned SiLU into a binary SFPU op. This removes a whole phase (FPU pass + CB round trip),
  not instructions inside one.
- The likely sibling child of r03-b02-a02 (port the shorter silu body to PACK) leaves multiply_phase in
  place, so the two are complementary.

## Expected effect and risk
- SiLU shapes: −5 µs per expert per row. dsv3/glm ≈ −2%, kimi-k2.7 ≈ −1.5..−2%, mistral ≈ −2..−3%
  (lottery-noisy).
  - PACK now does silu plus one multiply per tile (slightly more SFPU per subblock). That is still hidden
    behind the next subblock's matmul, except the last subblock: ~4 tiles of SFPU leak into DN, as on the
    binary shapes.
  - On DRAM-bound dsv3/k2.7 the gain may partly turn into DN waiting for down weights (r02-b01-a03 finding).
- Other shapes are unchanged (same code path). Expected geomean +0.5..1%.
- Risks:
  - Accuracy gate: the change makes the product more precise, so PCC / mean_abs should not get worse.
    If they do, it shows as accuracy_fail on the 4 SiLU shapes.
  - JIT error in the new body (static_assert, includes).
  - Binary size: the SiLU programs lose multiply_phase on MATH and gain a ~25-instruction body on PACK,
    so roughly neutral.
