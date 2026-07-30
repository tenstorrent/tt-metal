# coordinator_kernel.cpp — annotation

File: `tt_metal/programming_examples/contributed/multicast/kernels/dataflow/coordinator_kernel.cpp`
Role: SENDER (didactic, single-shot). The cleanest reference block in the group. INTRA-CHIP. IN SCOPE.

## THE BLOCK (sender half), lines 30–66
| step | lines | call |
|------|-------|------|
| set up source L1 (read tile DRAM→L1) | 31–32 | `noc_async_read` + `noc_async_read_barrier` |
| pre-handshake: wait for receivers ready | 45 | `noc_semaphore_wait(sender_addr_ptr, num_dests)` |
| reset handshake sem | 48 | `noc_semaphore_set(sender_addr_ptr, 0)` |
| build mcast addr (data) | 56–57 | `get_noc_multicast_addr(start_x,start_y,end_x,end_y, tile_l1_addr)` |
| **mcast DATA block** | 58 | `noc_async_write_multicast(tile_l1_addr, addr, single_tile_size, num_dests)` |
| set local flag VALID | 61 | `*(receiver_addr_ptr) = VALID` |
| build mcast addr (sem) | 62–63 | `get_noc_multicast_addr(..., receiver_addr)` |
| **mcast SEM flag** | 64 | `noc_semaphore_set_multicast(receiver_addr, addr, num_dests)` |
| flush/barrier | 66 | `noc_async_write_barrier()` |

## Forks
- **F1 = barrier** (line 66, `noc_async_write_barrier`).
- **F2 = flag** (level flag: local `=VALID`, mcast set; receiver waits `==VALID` then resets). Note the pre-handshake side (line 45/48) uses a **counter** semantics (`wait(sender_addr_ptr, num_dests)` then reset to 0) — so this kernel mixes F2: counter for the ready-collect, flag for the data-ready signal.
- **F3 = EXCLUDE_SRC**: coordinator core is the data source, not in the destination rectangle. No loopback. `num_dests` is the full receiver count.
- **pre_handshake = present** (lines 45,48): receivers signal readiness into `sender_addr` before the sender multicasts. Dest is fresh each call (single-shot here).

## HOLE flags
- Data mcast and sem mcast use **separate `get_noc_multicast_addr` calls** for the same rectangle (lines 57 vs 63) — Pipe should build the rectangle once and reuse.
- No NOC1 coordinate swap here (assumes noc 0). Receiver-grid coord-swap (seen in other kernels) is a HOLE the helper must own.
