## Files read
- agent_orch/WORKER.md, campaign.yaml — rules, metric, allowed paths
- tests/ttnn/nightly/unit_tests/operations/fused/test_routed_expert_prefill.py — 9 shapes, ROW_MAJOR
  dispatch buffer, bf4 ND-sharded weights, uniform top-k routing => 68-272 tokens/expert
- unified_routed_expert_ffn/device/unified_routed_expert_ffn_program_factory.cpp — 11x8 grid, M on
  rows / N on columns, CB sizing + L1 fitter, sender placement (weight sender row gx%8, x sender col
  (gy+1)%11), UP_SPLIT / DOWN_SPLIT / IN1_WRITER_MCAST semaphores
- kernels/adaptive_chunk.hpp — runtime per_core_M = ceil(tail/8) => 1 for all <=8-tile experts
- kernels/weight_runs.hpp — one NoC read per ND-shard row (per_core_N tiles)
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp — per-K-block serial read->mcast on sender
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp — writer up read + gate/up mcast serial;
  writer down half gated by reader's down_go at its iteration top
- kernels/compute/fused_swiglu.cpp — matmul_block rt=1, in1 popped once per K-block
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h — in1 (SrcA) unpacked once per
  output tile whenever ct_dim >= rt_dim, so per-core M=1 has no in1 reuse (compute ~23 cyc/tile)
- tt_metal/hw/inc/api/dataflow/circular_buffer.h, api/semaphore.h — pages_reservable_at_back,
  Semaphore::value() for the non-blocking checks
- hybrid_routed_expert_ffn/* (skimmed) — merged fused/unified op; not used by this test
- models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py — ND-shard spec (1 tile-row x per_core_N,
  round-robin over banks; ~370 GB/s aggregate rotating)
## Nodes consulted
- none exist yet (round 1, history empty)
## Docs / external references
- tests/.../deepseek_prefill/test_single_routed_expert_perf.py — single-expert time vs active tokens
  (marginal ~70 us per extra tile-row of M; M=1 costs 104 us => delivery-bound at M=1)
- tests/.../deepseek_prefill/test_moe_fused_swiglu_perf.py — the fused op at 128 tokens is ~84 us
- git log of the op (#55636, #56638) — earlier sender-placement / ND-shard work
- baseline profiler CSV ($DREAM_HOME/routed-expert-prefill/reports/baseline_1) — 88 cores, all RISCs
  busy for the full kernel duration (no per-zone markers)
