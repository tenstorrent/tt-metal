# Migration audit — matmul group

Pipe = two-sided object: `send()` (R→S wait + data-mcast + flush/barrier + flag-mcast) / `receive()` (set INVALID + signal-back + wait flag). Tag per kernel block.

| # | Kernel | Role | Tag | Notes / blocker |
|---|--------|------|-----|-----------------|
| 1 | reader_bmm_tile_layout_in0_sender_padding.cpp | sender | **clean** | Canonical send(). Extra flag-only mcast for sparsity batch-valid → needs `send()` data-less mode. EXCLUDE_SRC, flush, level flag, pre_handshake=yes. |
| 2 | reader_bmm_tile_layout_in1_sender_writer_padding.cpp | sender + writer | **clean** (reader half) | Two send() instances (in1, in3). Writer half is out-of-family, untouched. Cost: file co-residence only. |
| 3 | reader_bmm_tile_layout_in0_sender_dram_sharded.cpp | hybrid 3-core-type | **refactor** | One binary = sender-no-compute / sender+compute(INCLUDE_SRC loopback) / receiver. send() must do both EXCLUDE & INCLUDE_SRC; SKIP_MCAST is asymmetric (keeps sync, drops data); manual ping-pong source addr. Cost: medium — dispatch on worker_core_type, source addr not from cb.get_write_ptr(). |
| 4 | reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp | hybrid send/recv per-block, rotating sender | **refactor (high)** | The stress kernel. Same operand flips sender↔receiver by `block_id==sender_id`; loopback in-rect; unicast degenerate at num_cores==1; send-only-core CB lockstep pop; rotating sender coord table. A pure two-sided Pipe does not fit; needs a "self-rotating role" mode. Highest cost. |
| 5 | reader_mcast_transformer_group_attn_matmul.cpp | hybrid send/recv per tile_row, rotating sender | **refactor (high)** | Like #4 but per-32-row. **Uses BARRIER after flag (F1 minority) + per-iter barrier.** INCLUDE/EXCLUDE_SRC runtime select. Sems from runtime args. Cost high + F1 disagreement with the likely flush winner. |
| 6 | reader_bmm_tile_layout_in0_receiver.cpp | receiver | **clean** | Canonical receive(). One `wait_min` (sparsity) vs `wait` (steady) → receive() needs min-threshold mode + flag-value readout (3-state VALID/INVALID/IGNORE_BATCH). |
| 7 | reader_bmm_tile_layout_in1_receiver_writer_padding.cpp | receiver + writer | **clean** (reader half) | Two receive() instances. Writer out-of-family. |
| 8 | reader_bmm_tile_layout_in1_ring_all_gather.cpp | flag-only start barrier | **defer/raw** | `do_signaling()` is a one-shot collective GO barrier (counter gather + flag mcast), no data half. Different lifecycle from per-block Pipe. |
| 9 | reader_bmm_tile_layout_in0_ring_all_gather.cpp | ring unicast forward | **defer/raw** | No multicast at all. Unicast + monotone wait_min/up ring. Wrong helper entirely. |
| 10 | (pe) reader_bmm_tile_layout_in0_sender_in1_sender.cpp | sender ×2 (raw API) | **clean** | Textbook send(). Raw C API spelling. |
| 11 | (pe) reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp | receiver ×2 (raw API) | **clean** | Textbook receive(). |
| 12 | (pe) reader_bmm_tile_layout_in0_receiver_in1_sender.cpp | recv(in0)+send(in1) (raw) | **clean** | Two independent Pipes per kernel, distinct operands. |
| 13 | (pe) reader_bmm_tile_layout_in0_sender_in1_receiver.cpp | send(in0)+recv(in1) (raw) | **clean** | Mirror of #12. |

## Not block instances (naming false-positives, no mcast/handshake)
- reader_bmm_tile_layout_in1_sender_dram_sharded.cpp — "sender" = reads DRAM-sharded weights, no mcast.
- reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp — same.
- reader_bmm_tile_layout_in0_sender_dram_sharded_height.cpp — no relevant calls.

## Counts
- **clean: 8** (#1,2,6,7,10,11,12,13)
- **refactor: 3** (#3 medium, #4 high, #5 high)
- **defer/raw: 2** (#8 flag-only barrier, #9 ring unicast)
- Total block-bearing kernels: 13. False-positives excluded: 3.

## Headline blockers
1. **Rotating-sender hybrids (#4, #5)**: one binary where a core is sender for some blocks and receiver for others. Breaks "two distinct objects." Needs a self-role Pipe or stays raw.
2. **INCLUDE_SRC loopback (F3)** appears in #3,#4,#5 — `send()` must take a mode param (EXCLUDE_SRC default, INCLUDE_SRC when sender ∈ rect) and degenerate to unicast at num_cores==1 (hang otherwise).
3. **F1 disagreement**: #5 (and the do_signaling in #8) barrier-after-flag, while #1–4,10–13 flush-between + barrier-at-end. The bake-off must settle flush-vs-barrier; #5's per-iter barrier is a perf outlier.
4. **Flag overloading**: VALID/INVALID/IGNORE_BATCH (3-state) + wait vs wait_min (#1,#6). receive()/send() must treat the flag value as a payload and support a min-threshold predicate.
5. **Source addr not always cb.get_write_ptr()** (#3 manual ping-pong; #4 sharded-CB self-read) — `send()` source must be a free address arg.
