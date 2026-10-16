## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml — rules, metric, allowed paths.
- unified_routed_expert_ffn/device/kernels/compute/fused_swiglu.cpp (b01-a02 version) — fused gate/up
  K-loop (1 x w subblocks, PACKER_L1_ACC + per-block packer drain), row-shared tilize, bias pops at
  the end of each expert (matters for the FUSE_BIAS deadlock argument).
- .../kernels/dataflow/unified_routed_expert_ffn_reader.cpp — gate/up loop (reserve -> up_go ->
  acks -> gate read -> mcast_go -> x-owner multicast -> wait valids), down loop, per-expert bias
  reserve before the chunk loop, x up-front reads.
- .../kernels/dataflow/unified_routed_expert_ffn_writer.cpp — up feed + IN1_WRITER_MCAST, DOWN_SPLIT
  feed, cb_out drain; the feed of unit u+1 starts only after unit u's drain.
- .../kernels/adaptive_chunk.hpp — group-size pick, expert_cost, GroupScheduler (copyable, used for
  the next-unit lookahead).
- .../unified_routed_expert_ffn_program_factory.cpp — L1 fitter, subblock choice, CB footprint (to
  rule out deeper weight rings: v4p/mm3 have no room for a 3rd slot at per_core_M_max 4).
- tt-llk blackhole llk_unpack_AB_matmul.h — matmul_block unpack: the streamed operand is unpacked
  once per tile-mult whatever the subblock shape (unpacks = rt*ct + min(rt,ct) per call), so taller
  subblocks (suggested by b02/b03 reflections) do NOT cut unpack work; dropped that idea.
- tech_reports/Saturating_DRAM_bandwidth, models/experimental/voxtral_tts/README.md (BH p150b DRAM
  ceiling 367 GB/s measured), unified_routed_expert_ffn_device_operation.cpp (30 GB/s per-bank note).
- tt_metal/hw/inc/api/semaphore.h (value() is a cache-invalidated read), circular_buffer.h
  (pages_available_at_front for a non-blocking poll).
- reports/r01-b01-a02/.logs/profile_log_device.csv — per-core TRISC_1 end times per group per run:
  groups with equal work finish 3-9% apart (max/mean: mistral 1.09, dsv4-flash 1.064, dsv3 1.061,
  glm 1.056, mm3 1.058, kimi-k2.7 1.041, kimi-k3 1.031, dsv4-pro 1.025, gpt-oss 1.007).
## Nodes consulted
- r01-b01-a01 / r01-b01-a02 — row groups + row-shared tilize; the base of this node (code copied).
  a02's reflection item 3 (expert-boundary bubble) is what this node implements.
- r01-b02-a01 / r01-b02-a02 — confirms x reads must stay off weight-sender cores; no 1-row groups.
- r01-b03-a01, r01-b04-a01 — in-loop weight read-ahead regressed (root schedule); this node keeps the
  in-loop order and only moves the next unit's first block.
- r01-b03-a02 — x batching regressed; reverted (this node drops the parent's kernels).
- r01-b04-a02 — in0_block_w_gu is an accuracy knob (don't widen).
## Docs / external references
- tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md (trid-based multi-block
  in-flight reads; considered, not used here).
