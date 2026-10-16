## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml — rules, allowed paths, gate.
- unified_routed_expert_ffn/device/unified_routed_expert_ffn_program_factory.cpp — CB sizing
  (cb_footprint_bytes), L1 fitter (fit_config), the M-doubling step and its calibration note,
  CB allocations and ring checks; in0_block_w_gu / per_core_M_max are compile-time args to all
  three kernels.
- kernels/compute/fused_swiglu.cpp — per-K-block costs in matmul_phase_fused_gu: RM tilize with
  InitAndUninit + reconfigs + matmul_block_init each block, packer STALLWAIT drain each block;
  CB block sizes are compile-time max, runtime per_core_M bounds work only.
- kernels/adaptive_chunk.hpp — runtime chunk picker: one chunk whenever count_tiles <= 8*M_max.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp (gate/up K-loop) — per-block handshake
  and RM stick reads (rm_kblock_bytes = w*32*2), all parameterised by in0_block_w_gu.
- reports/baseline_1/run.log — per-shape counts_min/max (68-272 tokens per expert).
- tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h — L1 size for the budget estimate.
- /tmp/rnb04a02/fit.py (scratch, not committed) — Python replica of cb_footprint_bytes/fit_config
  to predict the chosen (per_core_M_max, in0_block_w_gu) per shape.
## Nodes consulted
- r01-b01-a01, r01-b02-a01 — row-group expert parallelism (wins, 1.44/1.40); orthogonal to this.
- r01-b03-a01, r01-b04-a01 — sender read-ahead, both regressed; reverted the parent's version
  here, and they point at per-block fixed cost rather than DRAM latency.
## Docs / external references
- none
