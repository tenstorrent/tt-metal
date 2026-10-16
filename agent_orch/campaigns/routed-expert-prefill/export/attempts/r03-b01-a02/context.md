## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml — rules, allowed paths.
- history.md — every node's mechanism / score / next.
- device/kernels/adaptive_chunk.hpp — plan_launch / LaunchPlan, the three-RISC planner.
- device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp — counts bcast, scratch CB push, plan call site.
- device/kernels/dataflow/unified_routed_expert_ffn_writer.cpp — counts wait, start read, plan call site.
- device/kernels/compute/fused_swiglu.cpp — UNPACK plan call site, expert scan + mailbox to MATH/PACK.
- device/unified_routed_expert_ffn_program_factory.cpp — CB index table, scratch CB sizing, L1 margin
  (48 KB for small scratch CBs), kernel descriptors / named compile-time args.
- tt_metal/hw/inc/api/dataflow/dataflow_api.h — cb waits invalidate the BH L1 data cache (fence).

## Nodes consulted
- r03-b01-a01 (parent) — failure: kernel config buffer overflow 512-2096 B on h7168; reflection's fix list.
- r03-b02-a01 — same combination made to fit by a planner trim, 1.5887 (best); diff vs parent is only
  adaptive_chunk.hpp + comments.
- r03-b03-a01, r03-b04-a01 — planner prologue cost measurements (RE-PLAN 2-20 µs on NCRISC).
- r02-b02-a04, r02-b03-a04, r02-b01-a04 — the two halves of the combination.
- All earlier rounds via history.md "next" columns (DM look-ahead / VCs / read-ahead are dead ends).

## Docs / external references
- Standalone size check: /tmp/rep-r03b01a02-4RZl/sz.cpp compiled with the eval's riscv-tt-elf-g++ (-mcpu=tt-bh):
  plan_launch ~3.6 KB at -Os, 9.2-11.2 KB at -O3.
