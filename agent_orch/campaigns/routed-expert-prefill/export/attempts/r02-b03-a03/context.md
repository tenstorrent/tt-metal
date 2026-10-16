## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml — rules, metric, allowed paths.
- device/kernels/compute/fused_swiglu.cpp — binary_activation_phase (copy gate/up to dst, binary
  SFPU op per output tile in chunks of 4, pack), init once at kernel start, matmul_block_init
  before every GU and DN phase (so the MATH replay region is re-programmed after each activation).
- ttnn/.../moe_gpt/device/kernels/swiglu_sfpu.h, tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/
  ckernel_sfpu_{situ_glu,clamped_silu_glu,exp,recip,sigmoid,softcap,tanh}.h, api/compute/{situ_glu,
  clamped_silu_glu}.h — the stock ops, their constants and inits (Prgm0..2 usage).
- tt-llk blackhole llk_math_eltwise_binary_sfpu.h (SfpuType only matters for int ops),
  cmath_common.h (replay_buf_offset 16: FPU/SFPU split), llk_math_matmul.h (matmul records its
  MOP into the replay buffer at init), llk_math_eltwise_unary_datacopy.h / eltwise_binary.h (no replay).
- tt_metal/jit_build/build.cpp, llrt/hal/tt-1xx/blackhole/bh_hal.cpp — JIT flags/includes, used to
  build a local sfpi harness (/tmp/r02b03a03_sfpu) and a mini kernel (/tmp/r02b03a03_k) to count
  instructions and compile-check all three TRISCs for all three variants.
- jitcache ELFs (r02-b04-a02's eval) disassembled: per-row loop of the stock ops.

## Nodes consulted
- All 24 nodes' reflections (history.md + git show of each).
- r02-b01-a02, r02-b04-a02 — MATH phase zones: activation cost per expert per shape (the target).
- r02-b04-a01 / r02-b02-a01 — gpt-oss F ~150 µs per expert fixed cost; activation is a large part.
- r02-b03-a02 (parent) — current schedule (two-phase + mixed layouts), kept unchanged.
- r01-b02-a04 — per-call LLK overhead is not the matmul bottleneck (so attack SFPU, not matmul).

## Docs / external references
- tt_metal/hw/ckernels/blackhole/.../ckernel_sfpu_tanh.h comments on sfpi replay of uniform bodies
  and LReg pressure.
