## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml — job, rules, allowed paths.
- $DREAM_HOME/routed-expert-prefill/history.md — index of all 48 committed nodes (rounds 1-3) and the round-4 root (r03-b02-a03).
- kernels/compute/fused_swiglu.cpp — gate/up K-loop (`matmul_phase_fused_gu`), down `matmul_phase`, kernel_main
  (hw startup, counts handshake, planner, per-expert loop, RE-GU/RE-ACT/RE-DN zones, RE-HW hooks). Where the probe and
  the wait tallies go.
- kernels/compute/re_hwprof.hpp — r03-b02-a03's MATH perf-counter windows and RE-HW record layout; extended here.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp — gate/up loop (x owner RM read / tilize / row multicast,
  IN1_WRITER_MCAST gate read, UP_SPLIT handshake, the sender's x(kb) wait at the end of each iteration), down loop.
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp — writer's up read and gate/up multicast (IN1_WRITER_MCAST),
  down NoC-1 half.
- kernels/adaptive_chunk.hpp — x_owner_col (owner never the row's weight sender), cost model.
- device/unified_routed_expert_ffn_program_factory.cpp — up_mode (UP_SPLIT), kWriterMcastsIn1 (on by default), ring
  depths (kDeepRingSlots = 3 when L1 allows), staggered sender row (group_base + gx % rows), ComputeConfig.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h, llk_math_matmul.h — stock matmul unpack replay
  (RDCFG/ADDDMAREG/STALLWAIT/WRCFG per streamed tile), per-tile SETC16 + MOP math.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_unpack_AB_custom_mm.h — CFGSHIFTMASK address stepping.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h — dest release zeroes the half (clear_dest default), so the
  probe's discarded products cannot leak into real subblocks.
- tt_metal/hw/inc/api/compute/matmul.h, reg_api.h, api/semaphore.h — matmul_block / init, dst API, Semaphore::value().
- tt_metal/soc_descriptors/blackhole_140_arch.yaml — DRAM channel endpoints (3 per channel, x = 0 and 9).
- tests/ttnn/nightly/unit_tests/operations/fused/test_routed_expert_prefill.py — shapes, ND-sharded bf4 weights
  (1 tile-row x per_core_N shards, round robin), token counts.

## Nodes consulted
- r03-b02-a03 (root) — RE-HW counters: ~31 cycles/mult at FPU 16.1, UNP0 busy 17.5/mult; its profile used for the
  wall-time breakdown (`/tmp/r04b03a01/ends.py`: chip time = slowest core's DN end + ~2 µs; GU start 4-9 µs, 22 on kimi-k3).
- r03-b01-a04, r03-b02-a04, r03-b03-a04 — operand swaps / reuse-B halved unpack work per mult, cycles/mult unchanged.
- r03-b04-a04 — down-weight read before the receivers' acks (the mechanism I first implemented for gate/up + down).
- r03-b03-a02 — reader/writer per-site DM counters: sender cores wait 3.0-3.8 µs per block for x(kb).
- r01-b03-a01, r01-b04-a01 — read-ahead issued before the current multicast regressed 6%.
- r04-b02-a01 (committed during this attempt, new best 1.6275) — implemented the same gate/up + down look-ahead I had
  started; its gate/up half was neutral. I discarded my reader change (kept at /tmp/r04b03a01/reader_lookahead_discarded.cpp)
  and pivoted to this measurement.
- r04-b01-a01 (committed during this attempt) — CFGSHIFTMASK unpack removed the UNPACK-thread stall, cycles/mult
  unchanged; concludes the SrcA bank behaves single-buffered. This node tests that against data starvation.
- r04-b04-a01 (in progress, read-only peek at its proposal) — decouples x / weight handshakes on receivers.

## Docs / external references
- tt-metal #58713 (per-call unpack protocol ~55 cycles; 1xN, N >= 4 blocks at the 16-cycle floor), #58714 (math thread
  floor 17.1 cycles/tile at LoFi 1x8; L1_TO_L1 1x8x32 17.2), #58712 (streamed-tile address update ~16 cycles of thread
  time), PR #58817 (row MOP / ttsync unpack forms) — Blackhole LLK matmul harness numbers.
- tt-metal PR #47885 (bh_dram_read): spread readers ~344 GB/s, bank-adjacent placement ~500 GB/s on p150; PR #51726:
  aggregate DRAM read plateaus ~340-410 GB/s from 16 cores up.
