## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml: the job, rules, allowed paths and accuracy gate.
- /localdev/nstamatovic/dream/routed-expert-prefill/history.md, plus every node's proposal.md / reflection.md / error.txt (all 32 nodes, via git show on the tags).
- kernels/adaptive_chunk.hpp: the whole planner. plan_launch -> make_plan (pick_group_rows, thresh loop, model_phase x2, phase-B candidates) -> plan_groups (3 uniform schedule_in_order + 3 mixed schedule_lpt + final). `tiles_of` is called ~47x per expert, and expert_cost / num_chunks divide.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp: the counts multicast, the `start` read + barrier, the tiles_of lambda, plan_launch, and the gate/up K-loop (x owner, weight sender) for the bottleneck analysis.
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp: the `start` read + barrier, then plan_launch.
- kernels/compute/fused_swiglu.cpp: the matmul phases, the UNPACK plan_launch and the expert mailbox.
- kernels/weight_runs.hpp: ND-shard weight read runs.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h and llk_math_matmul.h: the matmul unpack reuse rule (reuse_a = ct >= rt; the streamed srcA unpack carries RDCFG/STALLWAIT/WRCFG per tile) and 16 MVMUL per tile at LoFi.
- tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h: TRISC local memory is 4 KB and the stack minimum 192 B, so the new arrays must stay small.
- tt_metal/jit_build/build.cpp: kernels build with -std=c++17 and -Os by default, so no `requires` clauses.
- tech_reports/GEMM_FLOPS/GEMM_FLOPS.md: the LoFi ideal of 16 cycles per tile.

## Nodes consulted
- r02-b01-a04: measured first-GU-start of 29.2 µs on kimi-k3, 10-11 µs on 8-12-expert shapes and 5-6 µs on 4-expert shapes with the b03 planner (6.4 / 3.2-3.9 / 2.5 µs with the old picker). This is the target of this node.
- r02-b03-a02: +21 µs median shift on kimi-k3 and +10 on dsv4-flash from make_plan + plan_groups. Its next #1 suggested caching tiles_of and rejecting before modelling.
- r02-b02-a04 (the root): adds mixed phase-B scoring to make_plan. Its next #3 is the planner trim.
- r02-b03-a04 / r02-b01-a04 next items: planner trim untried.
- r02-b01-a02: the earlier prologue (counts + start) was ~1 µs, and the `start` fold was neutral. That is why this node only overlaps the start read instead of moving it.
- r02-b04-a01 note: "these RISC-Vs have no hardware divider". That is wrong for BH: -mcpu=tt-bh is rv32im and emits `divu`. Divides are still multi-cycle, and the profile shows ~900 of them on kimi-k3.
- The other round-3 branches (b01, b02) are porting the PACK-thread activation pipeline onto this root. I checked their worktrees and picked a different, orthogonal lever.
- Compute-efficiency analysis (not acted on, for children): gpt-oss pcM 3 -> 4 costs +30 µs per expert, i.e. ~16.7 cycles per tile-mult at the margin, which is the LoFi ideal. Fitting T = F + U·pcM to mistral (1 expert, pcM 3) and dsv4-flash (2 experts, pcM 2) gives U ≈ 25 µs and F ≈ 100 µs per expert. So the compute-bound shapes are dominated by a per-expert fixed cost, not by per-tile matmul rate.

## Docs / external references
- Tools built for this node (in /tmp/rep-r03b04a01-rO2J):
  - fuzz.cpp: old vs new plan_launch equivalence on the host.
  - sim/prog.cpp + link.ld + rvsim.py + prof.py: compile the planner for BH (-Os) and count instructions / L1 loads / divides per function on the test routing in an RV32IM + Zba/Zbb interpreter.
  - orig/: the root header.
