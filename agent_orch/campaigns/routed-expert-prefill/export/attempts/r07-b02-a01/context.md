## Files read
- agent_orch/WORKER.md, campaign.yaml, history.md — job, rules, index of all 65 nodes.
- device/kernels/dataflow/unified_routed_expert_ffn_writer.cpp (whole file) — UP_SPLIT up-tail read, IN1_WRITER_MCAST
  gate/up multicast on NoC 1 for every sender, DOWN_SPLIT tail read and r06-b04-a01's bottom-sender down multicast.
- device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp (phase loop, gate/up K-loop, down K-loop) — gate read +
  up head rows, look-ahead (try_gu_lookahead), the compiled-out reader gate/up multicast path, wmc_down, x-owner and
  activated row multicasts.
- device/kernels/adaptive_chunk.hpp — sender_row_in_group (top row when (c + base) even, else bottom), x_owner_key,
  layout tables (no 1-row groups).
- device/unified_routed_expert_ffn_program_factory.cpp (IN1_WRITER_MCAST block) — DS_NO_WRITER_MCAST fallback, NoC split.
- hybrid_routed_expert_ffn/device/kernels/hybrid_reader.cpp / hybrid_writer.cpp — they #include the unified dataflow
  kernels, so the change is shared (no HYB_* hooks in the unified kernels on this lineage).
- tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h — reads use NCRISC_RD_CMD_BUF on read_req_vc,
  multicasts the static multicast VC with the LINKED bit; so reads issued between a linked multicast and its valid sem
  use a different command buffer and VC.

## Nodes consulted
- r06-b04-a01 (parent, best 1.6931) — wrap-free down multicasts; next #1 = this node (gate/up half), option (a).
- r05-b04-a01 (hang) — wrap diagnosis and the gate/up wrap/no-wrap wait table; repair option (c) = switch the
  multicasting RISC, not the sender row.
- r05-b02-a01 — NoC 60/40 split, link model noc4.py (reused in /tmp/r07b02/m/gufix.py).
- r06-b01-a01, r06-b02-a01, r06-b03-a01 — row gradient is first-mover, not steerable by byte placement or start
  stagger; NoC-1 endpoint remap hurts gate/up (port concentration).
- r04-b02-a01 — gate/up look-ahead (kept and reused for the reader-multicast senders).
- r04-b04-a02 — [&] captures in noinline lambdas pin locals (~0.8 KB); here by-value captures and a local Noc.
- r01-b03-a01 / r01-b04-a01 — never issue the next read before the multicast; after it (before flush) is their advice.
- r04-b01-a02, r04-b03-a03 — kernel-config-buffer limits (70656 B), local -O2 compile underestimates by up to ~0.5 KB.
- All other r01-r05 reflections read for what is ruled out (handshake decoupling, VCs, rings, read-ahead, planner).
- r07-b01-a01 proposal (in flight, read-only) — plan-once + tie-pass port; this node avoids that axis.

## Docs / external references
- Local size harness: /tmp/r05b02a01/b.sh (reader/writer -O2 compile), /tmp/r07b02/sz/b.sh (writer source override).
  Parent vs this node, RM x, writer-mcast, up_head_rows 3: reader 17420 -> 17376 B (-44, after the hang repair), writer 8656 -> 8380 B (-276);
  profiler writer 12284 -> 12008. TILE, bias, DS_NO_WRITER_MCAST and legacy-up variants compile.
- tt_metal/hw/inc/internal/debug/sanitize.h `debug_sanitize_check_linked_transactions` — "Submitting a non-mcast
  transaction if there's a linked transaction on any cmd_buf will cause a deadlock": the cause of run 1's hang.
- Analysis: /tmp/r07b02/ana.py and /tmp/r04b01a02/rows.py on reports/{r06-b04-a01,r07-b02-a01}/.logs/profile_log_device.csv.
