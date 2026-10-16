## Files read
- agent_orch/WORKER.md, campaign.yaml, history.md — rules, shapes, all node summaries.
- All 52 nodes' proposal / reflection (dumped via git show); rounds 3-4 read in full, r01-r02 via reflections + history index.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp (r04-b02-a01 version) — the phase-4 loop: lock-step act ack at
  top of kb, binary act_valid, act sender in its own iteration kb, down look-ahead + spin re-check.
- device/unified_routed_expert_ffn_program_factory.cpp — L1 fit, gu/d ring depths (deep rings already 3 on some shapes),
  semaphore list (15 of 16 NUM_SEMAPHORES used), CB indices (c_0..c_20), reader defines.
- kernels/compute/fused_swiglu.cpp — down matmul_phase waits in0 (activated) then in1 per block, pops per block.
- kernels/compute/re_hwprof.hpp — profiler-only MATH counters (~1.5 KB TRISC1 in profiler builds).
- tt_metal/hw/inc/api/dataflow/{circular_buffer.h,dataflow_api.h,semaphore_dm_impl.h}, api/semaphore.h —
  pages_reservable_at_back semantics, set_multicast implementation (4 B non-posted mcast write).
- tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h, llrt/hal/.../bh_hal_tensix.cpp — L1 unreserved base
  (MEM_MAP_END 40960 + 69 KB kernel config) to estimate the per-shape L1 budget (~1.41 MB) and leftover per shape.
- hybrid_routed_expert_ffn/device/kernels/hybrid_reader.cpp — includes its OWN copy of the reader, so unaffected.

## Nodes consulted
- r04-b03-a01 — the measurement: compute waits 30-58 % of DN on the activated block; matmul itself is 21-25 cyc/mult.
- r04-b02-a01 — best node; its reader is the base here (down look-ahead + spin re-check), next #2 = this idea.
- r04-b04-a01 — parent; gate/up ack decoupling was null; reader dropped; size limit data (+1.55 KB overflowed by 240 B).
- r03-b04-a04 — down look-ahead origin; next #1 suggests the activated stream as the next chain.
- r01-b04-a03 — deeper rings with lock-step acks: neutral (no run-ahead possible).
- r03-b02-a03 — RE-HW counters (now opt-in to free TRISC1 space).
- r03-b01-a01 / r03-b02-a02 — kernel-config-buffer overflow precedent and size-repair approach.

## Docs / external references
- Local compile tooling adapted from /tmp/r04b04/cc/build.sh (reader) and /tmp/r03b02a03/sz/kc.sh (TRISC) into
  /tmp/r04b04a02/{cc,sz}.
