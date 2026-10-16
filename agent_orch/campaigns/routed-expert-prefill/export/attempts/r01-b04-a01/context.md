## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/routed-expert-prefill/campaign.yaml — job + gate.
- tests/ttnn/nightly/unit_tests/operations/fused/test_routed_expert_prefill.py — shapes: experts_per_chip 4-28, 74-272 tokens per expert (3-9 tile-rows), ROW_MAJOR bf16 dispatch buffer, bf4 ND-sharded weights (shard width = per_core_N, height 1 tile-row, round-robin over DRAM banks).
- models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py — weight memory config, LoFi config.
- unified_routed_expert_ffn.cpp / device/unified_routed_expert_ffn_types.hpp — op entry, 11x8 fixed grid.
- device/unified_routed_expert_ffn_program_factory.cpp — CB sizing/L1 fit (per_core_M_max 4, in0_block_w_gu 16), sender placement (weight sender row gx%8, x sender col (gy+1)%11), UP_SPLIT / DOWN_SPLIT / IN1_WRITER_MCAST modes, sems.
- device/kernels/adaptive_chunk.hpp — runtime per_core_M: every benchmark expert is one chunk with per_core_M = ceil(count_tiles/8) = 1 (2 for >256-token gpt-oss experts).
- device/kernels/weight_runs.hpp — one NoC transaction per K-row shard slice.
- device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp — gate/up loop (gate read on NoC0, handoff to writer), down loop (reader reads lower half + multicasts, waits in1_ready BEFORE reading).
- device/kernels/dataflow/unified_routed_expert_ffn_writer.cpp — per K-block: up read -> barrier -> multicast gate+up, strictly serial; down upper-half reads; output drain.
- device/kernels/compute/fused_swiglu.cpp — PACKER_L1_ACC matmul phases, tilize per K-block, subblock h=1.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h — matmul streams one operand per output tile (reuse only along one dim), so per-core unpack count ~ tile-mult count regardless of M/N split; rules out an "in1 reuse" layout change as a cheap compute win.
- tt_metal/hw/inc/api/semaphore.h, api/dataflow/circular_buffer.h, dataflow_api.h, internal/tt-1xx/blackhole/stream_io_map.h — Semaphore::value(), pages_reservable_at_back, tiles_acked stream register (core-global, readable from any DM RISC).
- $DREAM_HOME/routed-expert-prefill/reports/baseline_1 — per-shape times, run.log probe lines (token counts), ops CSV (no per-RISC breakdown beyond equal kernel durations).
## Nodes consulted
- none exist yet (round 1; history.md lists 0 attempts).
## Docs / external references
- none.
