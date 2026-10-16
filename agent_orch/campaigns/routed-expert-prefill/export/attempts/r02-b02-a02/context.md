## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/routed-expert-prefill/campaign.yaml: rules, allowed paths, accuracy gate.
- kernels/adaptive_chunk.hpp: layout table (10 / 5+5 / 2x5 / 3+3+2+2 / 3+3+3), expert_cost = units + chunks, LayoutScheduler.
  Replayed the picker on the test routing with F (per-chunk fixed cost) = 1..5 (/tmp/routed_r02b02a02/pick.py): F >= 2 only moves
  dsv4-pro back to 2x5, so recalibrating F on this lineage is near-useless.
- kernels/compute/fused_swiglu.cpp: gate/up K-loop with PACKER_L1_ACC, binary activation phase on MATH SFPU, down matmul.
  The activation is serial between gate/up and down, and the down phase needs column kb's activation for block kb, so
  within-expert SFPU/FPU overlap can only hide about one K-block's work.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp (phase 4 down feed, gate reads), writer (up / down-split reads),
  kernels/weight_runs.hpp: every weight read goes through WeightRuns::read with the default request VC (1).
- tt_metal/hw/inc/api/dataflow/noc.h: async_read<NocOptions::CUSTOM_VC> + NocOptVals{.vc}; default req VC = NOC_UNICAST_WRITE_VC.
- ttnn/.../matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:728: vc = bank_id & 3,
  bumped so that readers in one row differ.
- tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md: FCFS within a VC lets one reader starve another
  in the same row, and distinct VCs fix it.
- tt_metal/core_descriptors/blackhole_140_arch.yaml + profiler header "Max Compute Cores: 120": the device is 2x harvested,
  so the compute grid is 11x10 (the 12th column is dispatch) and there are no spare columns to widen into.
- models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py: weights ND-sharded, shard = 1 K-row x per_core_N, round-robin
  over banks.
- ttnn/.../moe_gpt/device/kernels/compute.cpp + swiglu_sfpu.h: PACK-thread SFPU pattern (considered and rejected for this node).
- hybrid_routed_expert_ffn kernels include unified's weight_runs.hpp, so the new vc argument defaults to the old VC.

## Nodes consulted
- All 20 nodes (proposal/score/reflection). Key ones:
- r02-b02-a01 (parent), r02-b04-a01: mixed layouts; per-expert fixed cost F; few-expert shapes are not DRAM-bound.
- r01-b01-a02, r01-b03-a03, r01-b04-a03: groups doing identical work end 6-15% apart, the slow group changes per call
  and is slow for the whole call, and deeper rings don't absorb it.
- r01-b01-a03: pacing equalised the groups but cost 30% through critical-path overhead.
- r01-b02-a04 reflection #3: proposed the request-VC probe (untried until now).
- r02-b01-a01, r02-b03-a01: per-group stream rate limits few-group layouts; big shapes sit at ~410-440 GB/s.
- Parent profile (reports/r02-b02-a01/.logs/profile_log_device.csv, /tmp/routed_r02b02a02/grp.py, risc.py): per-call group
  ends on mm3 range 310-434 µs for ~equal work, and mistral 157-202.
