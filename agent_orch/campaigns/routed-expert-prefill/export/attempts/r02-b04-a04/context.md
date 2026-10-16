## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml — rules, gate, allowed paths.
- device/kernels/compute/fused_swiglu.cpp — SiLU gate copy pass (silu_tile per dst tile on MATH after the GU K-loop), BINARY_ACT macros, kernel init.
- device/kernels/weight_runs.hpp, models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py — weight ND-shard layout (1 tile-row x per_core_N shards, round-robin over banks); ruled out a bank-rotation probe as low value.
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h, ckernel_sfpu_sigmoid.h, ckernel_sfpu_exp.h, ckernel_sfpu_recip.h — stock silu = x * recip_iter<1>(1 + exp_21f<true>(-x)), bf16 RN; recip init sets Prgm0 = 2.0.
- tt_metal/hw/inc/api/compute/compute_kernel_api.h, llk_math_eltwise_unary_sfpu_macros.h — silu_tile / SFPU_UNARY_CALL plumbing.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib — which MATH LLKs use replay (matmul, reduce, transpose_dest, welfords; not eltwise binary / datacopy).
## Nodes consulted
- All 28 nodes' proposal/reflection (history.md + git show).
- r02-b03-a03 — the op-local binary SFPU header ported here; its next #1 (SiLU untouched).
- r02-b01-a02 / r02-b04-a02 — activation zone measurements.
- r02-b01-a03 — PACK-thread pipelined activation (SiLU gains small because silu is cheap per tile).
- r02-b04-a01 / r02-b04-a03 — this lineage's layouts; DRAM floors per shape.
- r02-b02-a03, r02-b02-a02, r01-b04-a03 — DM timing / VC / ring-depth levers ruled out.
## Docs / external references
- none
