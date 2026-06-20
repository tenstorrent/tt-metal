# Tier plan — mcast_pipe rollout re-entry @ v7 (2026-06-20)

Mode: `run-all`. Scope (per user): **Tier 0 only — remigrate the 13 stale kernels**. The 48 `pending`
and 8 `deferred` are untouched.

## Why this is a real remigration (not an arg-reorder)
The 13 migrated kernels are frozen at **v4** and use the **old single-object `dataflow_kernel_lib::Pipe<>`
API, which no longer exists in v7** (v7 = split `SenderPipe`/`ReceiverPipe`). The v5→v6→v7 handoffs in the
changelog were never executed. Confirmed at the compiler: `error: 'Pipe' is not a member of
'dataflow_kernel_lib'` + `'McastRect' used without template arguments`. Every stale kernel fails JIT
compile against v7 — the tree is red until remigrated.

## old `Pipe<>` (v4)  →  v7 `SenderPipe` / `ReceiverPipe` translation cheat-sheet
- v4 took sem ids + recipient count + rect as **runtime ctor args**; v7 takes them as **compile-time
  TEMPLATE params** (all already `get_compile_time_arg_val` in these kernels → feasible).
- **Sender:** `Pipe<>(noc, McastRect{x0,y0,x1,y1}, num_dests, data_ready_sem, consumed_sem)` + `.send(s,d,sz)`
  → `SenderPipe<NOC_ID, DATA_READY_SEM_ID, NUM_ACTIVE_RECEIVER_CORES, PRE_HANDSHAKE, CONSUMER_READY_SEM_ID,
  DataReadySignal>(noc, McastRect<NOC_ID>{x0,y0,x1,y1})` + `.send(s,d,sz)`. Drop the manual
  `data_ready_sem.set(VALID)` before the loop — the ctor does it.
- **Receiver:** `Pipe<>(noc, McastRect::single_core(sx,sy), 1, data_ready_sem, consumed_sem)` + `.receive()`
  → `ReceiverPipe<DATA_READY_SEM_ID, PRE_HANDSHAKE, CONSUMER_READY_SEM_ID>(noc)` + `.receive(sx, sy)`
  (sender coords move from the ctor rect to the `receive()` call).
- **Control-only:** `.send_signal(VALID)` → `.send_signal()` (no arg). A pure-signal sender that never gates
  must declare `PRE_HANDSHAKE=false` (otherwise the new `static_assert` fires for a missing consumer sem).
- **No-handshake sites** (`PRE_HANDSHAKE=false`): now **omit** `CONSUMER_READY_SEM_ID` entirely (trailing
  default `UNUSED_SEM_ID`). Two such sites: ln-sharded `phase1_pipe`, conv-WS `act_mcast_pipe`.

## Co-compilation rule (load-bearing for validation order)
A validation test JIT-compiles **all** kernels in its program. If a program contains >1 stale kernel, the
test stays red until **every** stale sibling is migrated. `pending` siblings use raw open-coded primitives
(not the helper) → they compile fine and do not block. So: **migrate a family's full stale set, then run
the family's validation nodeid(s).**

## Tier 0 — the 13 stale kernels, grouped by family (order ≈ easiest/safest first)

### G1 — groupnorm (2 kernels, both receivers; gn_v2 sender is pending=raw, does not block)
1. `reader_mcast_receiver_unary_sharded_gn_v2.cpp` (clean, receiver)
2. `welford_reader_mcast_receiver_unary_sharded_gn_v2.cpp` (refactor, receiver)
- Validate: `test_group_norm.py::test_group_norm_with_block_sharded_v2_8x4_grid[...]` (high)

### G2 — topk (1 kernel; control-only signal)
3. `reader_final_topk.cpp` (clean, receiver/control) — `send_signal()`/`receive_signal()`; PRE_HANDSHAKE=false on any pure-signal sender.
- Validate: `test_topk.py::test_topk[sub_core_grids=None-largest=True-sorted=True-...]` (high)

### G3 — conv (4 kernels; HS sender is pending=raw, does not block)
4. `reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` (clean, receiver) — conv HS
5. `writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` (clean, sender) — conv BS
6. `writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` (clean, receiver) — conv BS (co-compiled with #5)
7. `activation_reader_width_sharded.cpp` (refactor, hybrid) — conv WS; **PRE_HANDSHAKE=false** (omit consumer sem)
- Validate: conv HS nodeid (#4), conv BS nodeid (#5+#6 together), conv WS nodeid (#7) — all high.

### G4 — matmul (5 kernels, the spine; all stale & entangled across matmul programs)
8.  `reader_bmm_tile_layout_in0_sender_padding.cpp` (clean, sender)
9.  `reader_bmm_tile_layout_in0_receiver.cpp` (clean, receiver)
10. `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` (clean, hybrid)
11. `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` (clean, receiver)
12. `reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` (refactor, role-flip)
- Validate AFTER all 5 migrated: matmul 1D interleaved nodeid (exercises #8,#9,#10) + matmul 2D in0-sharded
  nodeid (exercises #11,#12 and in1 siblings). Both high.

### G5 — layernorm (1 kernel; sharded LN sender, ln receiver is pending=raw, does not block)
13. `reader_mcast_sender_unary_sharded_ln.cpp` (refactor, sender) — phase1 `send_signal()` is **PRE_HANDSHAKE=false**; phase2 is a normal handshake `send()`.
- Validate: `test_layer_norm_sharded.py::test_layer_norm_sharded_single_stage[dtype=torch.bfloat16-...]` (high)

## Coverage
All 13 now have high-confidence device-verified validation cases (block_sharded device-verified during this
re-entry's Phase 1). **No coverage gaps** in the Tier-0 set.
