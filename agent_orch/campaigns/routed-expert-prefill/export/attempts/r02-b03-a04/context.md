## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml — rules, metric, allowed paths.
- device/kernels/compute/fused_swiglu.cpp (this lineage) — matmul_phase_fused_gu, binary_activation_phase
  (MATH-thread op-local SFPU), multiply_phase, two-phase kernel_main mailbox.
- r02-b01-a03's fused_swiglu.cpp (git show) — the pipelined last-block / PACK-thread SFPU design; its
  diff vs r02-b01-a02 applied 3-way (matmul_phase_fused_gu is identical between r02-b01-a02 and HEAD).
- device/kernels/compute/binary_act_sfpu.h — op-local SwiGLU-OAI / SiTU-GLU / clamped SiLU-GLU bodies,
  binary_act / binary_act_init entry points.
- ttnn/.../matmul/.../bmm_fused_activation.hpp — apply_activation_from_pack pattern (SEMWAIT on
  MATH_PACK, SETC16 packer dest offset, SFPU, STALLWAIT WAIT_SFPU).
- tt-llk blackhole llk_pack.h / cpack_common.h — the packer uses the replay buffer only in
  PackMode::Tilize (re-recorded at each tilize init); tt_metal/hw/inc/api/compute/tilize.h (Tilize pack mode
  armed by tilize_init, restored on uninit); api/compute/experimental/face_compressed_mm.h ("each thread
  ... its replay buffer": replay buffers are per thread).
- llk_math_eltwise_binary_sfpu_macros.h — SFPU_BINARY_CALL expansion (same call r02-b01-a03 ran on PACK).
- tt_metal/jit_build/build.cpp — KERNEL_COMPILE_TIME_ARGS define and -include of the named-CT-arg header,
  used to build a local compile harness (/tmp/r02b03a04_k, derived from r02-b03-a03's /tmp/r02b03a03_k)
  that compiled all three TRISCs for SiLU / SwiGLU-OAI+bias / SiTU-GLU / clamped SiLU-GLU, RM and TILE x.
- unified_routed_expert_ffn_program_factory.cpp — compute CT arg order (for the harness).

## Nodes consulted
- All 28 nodes' reflections (history.md + git show of each).
- r02-b01-a03 — the pipelined pack-thread activation (+2.6%), its per-shape gains and its next #1
  (combine with the mixed-height layouts).
- r02-b03-a03 (parent) — op-local SFPU bodies; its notes on sfpi replay and Prgm CRegs.
- r02-b01-a02 / r02-b04-a02 — activation cost per expert (zones).
- r02-b02-a03, r02-b04-a02 — DM-side activation-gap ideas that lost (not repeated).
- r02-b04-a03 — caution that PACK-thread hiding is bounded by dst half-sync.

## Docs / external references
- none beyond the repo sources above.
