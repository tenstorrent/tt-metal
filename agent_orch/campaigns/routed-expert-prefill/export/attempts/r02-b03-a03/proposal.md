# r02-b03-a03: same-math binary SFPU activations in fewer instructions: op-local copies of SwiGLU-OAI / SiTU-GLU / clamped SiLU-GLU that hoist fp32 constants into LRegs / spare CRegs, clamp via SFPSWAP min/max, and unroll so sfpi replays the body

## Motivation
The activation phase between gate/up and down is serial MATH-thread SFPU work. r02-b01-a02 and
r02-b04-a02 measured it with MATH zones: ~55 µs per expert on gpt-oss (~21% of a core's busy time),
~40 µs per expert on kimi-k3 (6 experts on the critical group, ~15%), ~39 µs on minimax-m3 (~10%),
~15 µs per expert on dsv4-flash/pro. gpt-oss and minimax-m3 are compute-bound (one expert per group,
~200-330 GB/s, far from the DRAM ceiling), so their activation time is on the critical path 1:1.

Disassembly of the JIT-compiled TRISC1 ELFs (jitcache, r02-b04-a02) and a local sfpi build of the
stock ops show that the per-row SFPU loop is dominated by constant re-loads:
- SwiGLU-OAI (moe_gpt swiglu_sfpu.h): 54 instructions per row, 18 of them SFPLOADI (fp32
  constants: alpha, 1/ln2, the three exp_21f polynomial coefficients reload as 2 SFPLOADIs on every
  row) and the three v_if clamps take 12 instructions.
- SiTU-GLU: 77 per row, 25 SFPLOADI (three tanh coefficients are re-loaded twice per row).
- Clamped SiLU-GLU: 44 per row, 13 SFPLOADI.
Measured ~1.9k (SwiGLU) / ~3k (SiTU) cycles per tile = ~60 / ~94 cycles per row, i.e. about one
cycle per issued instruction: the phase is instruction-issue bound, so fewer instructions per row
means a proportionally shorter activation phase.

## Mechanism
New op-local header `device/kernels/compute/binary_act_sfpu.h` (copy of the three stock ops, per
the campaign rule for shared helpers) used by `fused_swiglu.cpp` instead of the stock includes:
- Same arithmetic in the same order on the same fp32 values: exp_21f (Moroz) with the same clamp
  to [0,255], the same 2nd-degree polynomial, the same Newton reciprocal (1 iteration), the same
  tanh polynomial, the same bf16 round-to-nearest of the result. Only where the constants live
  changes:
  - SwiGLU-OAI / clamped SiLU-GLU: 1/ln2 and EXP_21F_BF16_C2 go to vConstFloatPrgm1/2 at init
    (Prgm0 stays 2.0 for the reciprocal); c0, c1 (and alpha) are hoisted into LRegs once per face.
  - SiTU-GLU: tanh_init owns Prgm0..2, so the other three tanh coefficients are hoisted into LRegs,
    and `up` is loaded after the gate half so it holds no LReg meanwhile.
- SwiGLU-OAI v_if clamps -> sfpi::min / sfpi::clamp (SFPSWAP), identical for every non-NaN value.
- `#pragma GCC unroll 8` on the SwiGLU and clamped bodies: at <= 32 instructions they fit the
  replay buffer, so sfpi records the body once per face and replays it (TRISC1 issues ~4
  instructions per row). The MATH replay region it records over (slots 0-31) is re-programmed by
  matmul_block_init before every gate/up and down phase, and copy_tile / add_bcast_rows use no
  replay, so nothing in this kernel depends on the overwritten slots.
- The fp32-dest path keeps calling the stock functions.
Per row (local sfpi build): SwiGLU 54 -> 37 (31 SFPU, replayed), SiTU 77 -> 65, clamped 44 -> 35.
Kernel-only change; no host rebuild.

## Why this is not a repeat
No node has touched the activation implementation. r02-b01-a02 / r02-b04-a02 measured it and
proposed hiding it (PACK-thread SFPU, cross-expert overlap); this attempt instead shrinks it
without any scheduling change, and is orthogonal to the schedule work of this branch (planner,
two-phase, mixed layouts), which it keeps unchanged.

## Expected effect and risk
- Accuracy: bit-identical to the parent on every shape (PCC / max_abs / mean_abs equal to every
  printed digit). Any digit change would mean a constant or operation order differs.
- gpt-oss: activation ~55 -> ~35-40 µs per expert on the critical path: -5..-7%. minimax-m3 -2..-4%.
  kimi-k3 SiTU -16% of ~240 µs on the critical group: up to -2..-3% (partly DRAM-bound).
  dsv4-flash / dsv4-pro -1..-2%. SiLU shapes (dsv3, kimi-k2.7, glm, mistral) unchanged (their
  activation is silu_tile in the GU copy pass, not touched).
- Risks: JIT compile error (sfpi register allocation; checked locally with the same compiler), and
  replay-buffer interaction (argued above). A hang or garbage output would point at replay.
