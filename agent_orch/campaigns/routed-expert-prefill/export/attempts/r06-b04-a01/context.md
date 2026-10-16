## Files read
- `agent_orch/WORKER.md`, `campaign.yaml` — job, allowed paths, accuracy gate.
- `device/unified_routed_expert_ffn_program_factory.cpp` — semaphores (15 of 16 used, so no new ones: the down
  multicast reuses mcast_go/done with one running sequence), IN1_WRITER_MCAST / DOWN_SPLIT setup, d_head /
  up_head split, per-layout group topologies (sender row = `group_base + gx % rows` before this node), the TILE
  x-sender collision TT_FATAL.
- `kernels/dataflow/unified_routed_expert_ffn_reader.cpp` — gate/up loop (writer multicast handshake, gu
  look-ahead, x owners), phase-4 down loop (block-0 read after acks, top-of-block read, step 2/3 multicast,
  try_down_lookahead with the act-spin re-check), the writer-lag drain before phase 4.
- `kernels/dataflow/unified_routed_expert_ffn_writer.cpp` — UP_SPLIT read + NoC-1 gate/up multicast (swapped
  rectangle corners), DOWN_SPLIT tail read, cb_out drain order.
- `kernels/adaptive_chunk.hpp` — layout tables (10-row: 10, 5+5, 5x2, 3+3+2+2, 4+3+3, 4+2+2+2), layout_group,
  x_owner_slots/rank/col (closed form for "skip columns c % rows == my_mt"), planner opt pragmas.
- `kernels/compute/fused_swiglu.cpp` — where compute derives x_slots / x_rank (all TRISCs) from group_word.

## Nodes consulted
- r05-b04-a01 — the wrap diagnosis (wrap vs no-wrap weight waits, column parity flips between phases) and its
  hang; its option (c) "switch the multicasting RISC per sender position" is what this node applies to down.
- r05-b02-a01 (parent) — NoC link model (/tmp/r05b02a01/noc*.py), NoC 1 vs NoC 0 link loads; extended in
  /tmp/r06b04/nocw2.py for this placement / NoC choice.
- r05-b01-a01, r05-b03-a01 — other round-5 directions (plan-once, owner stagger) to avoid repeating.
- r04-b01-a02 — the writer-only -Os planner pragma (measured: first-GU-start unchanged), ported for size.
- r04-b02-a01, r03-b04-a04 — down look-ahead, block-0 ordering rationale (kept).
- r04-b03-a01/a02, r04-b04-a01/a02, r04-b02-a02 — handshake-decoupling nulls (don't retry); conserved GU+DN.
- r03-b01-a01 — kernel-config overflow precedent (70656 B limit on h7168).
- r02-b04-a02 — writer-owned free-running down stream (-10..20%); this node keeps every read's timing.
- History index for all others.

## Docs / external references
- BH NoC: NoC 0 routes +x then +y, NoC 1 -y then -x (torus); multicast reaches the rectangle start corner
  first (model in /tmp/r05b02a01/noc4.py).
- Local size checks: /tmp/r05b02a01/b.sh (reader/writer), /tmp/r06b04/sz/kc.sh (3 TRISCs) against
  wt/r05-b02 as base. Reader +420 B, writer -364 B (with -Os planner), TRISC0 +368 (an inlining flip of
  llk_unpack_reconfig_data_format_srca, kernel_main itself -28), TRISC1 -60, TRISC2 +8.
- /tmp/r06b04/chk2.py: placement / x-owner consistency check over all 10- and 8-row layouts.
