## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/routed-expert-prefill/campaign.yaml: rules, metric, allowed paths.
- device/kernels/adaptive_chunk.hpp: chunk picker, `expert_cost`, `GroupScheduler`, `pick_group_rows`
  (uniform 10/5/2 layouts, in-order list scheduling). This is what the node generalizes.
- device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp / _writer.cpp: where the group size,
  my_mt, chunk_M, topology set and the per-expert `mine` test come from (all derived from group_rows).
- device/kernels/compute/fused_swiglu.cpp: UNPACK plans and mailboxes group_rows + per-expert counts
  to MATH/PACK; matmul_phase / matmul_phase_fused_gu / binary_activation_phase structure (checked the
  activation and copy-pass costs as alternative levers).
- device/unified_routed_expert_ffn_program_factory.cpp: GRID_X/GRID_Y, L1 fitter, subblock choice,
  per-core runtime args, per-group-size weight-multicast topology sets, sender-collision check.
- device/unified_routed_expert_ffn_types.hpp: kCoreGridX=11 is pinned (weights' ND shard width =
  ceil(N/11)), so using the 12th Blackhole column is not an option without changing the model contract.
- models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py (routed_expert_weight_memory_config): shard
  width tied to UNIFIED_ROUTED_EXPERT_CORE_GRID.x.
- tests/.../test_routed_expert_prefill.py: routing (seed 0, uniform top-k), shapes; replicated the
  per-expert tile counts in /tmp/r02b04a01/sim.py and plan.py.
- moe_gpt/device/kernels/compute.cpp + swiglu_sfpu.h: PACK-thread SFPU pattern (considered for
  hiding the activation; rejected: one K-block of FPU work can only hide ~1/6-1/2 of the activation).
- tt_metal/hw/ckernels/blackhole/.../ckernel_sfpu_situ_glu.h, ckernel_sfpu_silu.h: SFPU cost estimate.
- tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp: RISC-V ISA is rv32im_zmmul (no hardware divide), so
  the runtime schedule uses a division-free cost path.
- tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h: TRISC local memory 4 KB (owner[] on stack ok).
- reports/r01-b04-a04/.logs/profile_log_device.csv: epilogue after the last math end is ~1.5 us on
  every shape (not a lever); per-shape DRAM traffic computed from shapes/counts: dsv3 ~435, kimi-k2.7
  ~438, dsv4-pro ~420, glm ~412, kimi-k3 ~404, dsv4-flash ~390, minimax ~367, mistral ~352, gpt-oss
  ~200 GB/s -> only the last five have real headroom.

## Nodes consulted
- All 16 round-1 nodes (proposal + reflection + score). Most relevant:
- r01-b01-a01 / r01-b02-a01: row groups (the 1.44x win); cost model units + chunks.
- r01-b03-a04 / r01-b04-a04: 10-row uniform layouts, +2-3%; both list "mixed group heights /
  cost-aware layouts" as the next scheduling step; a tie keeps the earlier layout.
- r01-b04-a02: in0_block_w_gu is an accuracy knob (not touched here).
- r01-b02-a04, r01-b03-a03, r01-b04-a03: subblock shape, ring depth, unit-boundary feeds are exhausted.
- r01-b01-a03: pacing (don't); r01-b03-a01 / r01-b04-a01: read-ahead (don't).

## Docs / external references
- tech_reports/matrix_engine/matrix_engine.md: LoFi 8x16x16 per cycle -> 16 cycles per tile-mult.
