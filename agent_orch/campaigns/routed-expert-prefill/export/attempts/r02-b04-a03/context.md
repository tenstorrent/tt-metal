## Files read
- agent_orch/WORKER.md, campaign.yaml, history.md: rules, metric and the node index.
- device/kernels/compute/fused_swiglu.cpp: phase structure (fused GU K-loop with packer L1_ACC, binary
  activation on the MATH thread, down matmul). I studied it to judge whether the activation could
  overlap FPU work via PACK-thread SFPU (moe_gpt style). Conclusion: the dst half-sync limits the
  overlap to ~one matmul subblock per SFPU step, and single-expert groups have no independent FPU
  work. I estimated ~+1% for a large reader/writer/compute refactor, so I didn't do it.
- tt_metal/hw/ckernels/blackhole/.../ckernel_sfpu_situ_glu.h, ckernel_sfpu_clamped_silu_glu.h,
  moe_gpt swiglu_sfpu.h: the SFPU activations. They are already tight (~45-60 SFPU instructions
  per row iteration, matching the 1.9-3k cycles/tile measured); no bit-identical speedup is visible.
- device/kernels/adaptive_chunk.hpp: the layout table, the cost model (group_cost =
  ceil(t/rows) + 1 per chunk) and plan_groups (uniform in-order, mixed LPT, strictly-better rule).
- device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp: every in1 multicast is already
  guarded by `in1_num_receivers > 0` (GRID_Y == 1 legacy). The row-major x source is the rotating
  owner, so is_in0_sender is unused on that path.
- device/kernels/dataflow/unified_routed_expert_ffn_writer.cpp: the IN1_WRITER_MCAST gate/up
  multicast had no 0-receiver guard, so I added one.
- device/unified_routed_expert_ffn_program_factory.cpp: L1 fit (gpt-oss in0_block_w_gu 15, 6
  K-blocks), the kMinGroupRows mirror in num_chunks, the per-layout topology builder and its
  x/weight sender-collision TT_FATAL, and the kernel defines.

## Nodes consulted
- All 24 reflections (dumped with git show into /tmp/r02b04a03/refl.md).
- r02-b04-a02 (parent): regression. Its zones (RE-GU/ACT/DOWN) were reused for analysis; its code is
  reverted here.
- r02-b04-a01: base code restored. Its plan.py / cmp.py harness (/tmp/r02b04a01) was reused to
  replay the planner.
- r01-b02-a01 / r01-b02-a02 / r01-b01-a04: earlier 1-row-group evidence on the 8-row grid (kimi-k3
  tie at equal critical tile-rows) and the TILE-path x-sender collision that retired them.
- r02-b01-a02, r02-b04-a02: per-phase zones. ACT is 15-21% on kimi-k3 / gpt-oss; DN is
  proportional to its MAC share.
- r02-b03-a01/a02: per-group stream-rate and DRAM-floor numbers (~440-455 GB/s for 5 streams).

## Analysis done here
- /tmp/r02b04a03/lot.py on reports/r02-b04-a02 profile: per-row GU/ACT/DN per call. The group-tail
  lottery lives in the GATE/UP phase (e.g. mistral slow groups GU 168-173 vs ~110 µs, DN shorter).
  Sometimes one row of a 2-row group is slow alone (x-stream delivery). On gpt-oss the 2-row
  (pcM 4) groups' GU is a steady 125 µs (compute-bound), while the 3-row (pcM 3) groups take
  115-164 µs: delivery-bound when all groups stream gate/up at once.
- Test routing tiles (seed 0): kimi-k3 = 11 x 4 + 17 x 3 tile-row experts (95). The 1-row layout
  is modelled at 13 units vs 18 for 2x5; no other benchmark shape changes plan.
- C++ planner (new header, -DUNIFIED_RE_SINGLE_ROW_GROUPS=0/1, grids 10/8) vs the Python replay
  on 24k random count vectors: 0 mismatches (/tmp/r02b04a03/cmp.py).

## Docs / external references
- tt_metal/hw/inc/api/compute/situ_glu.h (API wrapper), bmm_fused_activation.hpp
  (apply_activation_from_pack pattern), moe_gpt compute.cpp (PACK-thread swiglu).
