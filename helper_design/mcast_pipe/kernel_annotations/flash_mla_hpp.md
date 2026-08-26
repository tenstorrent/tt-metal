# flash_mla.hpp (deepseek_v3_b1 unified_kernels) — annotation

File: `models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp`
Role: SENDER+RECEIVER, BRISC. K-chunk mcast block inside the per-chunk loop. INTRA-CHIP. IN SCOPE.

## THE BLOCK (mcast sender, inside k_chunk loop), lines 308–367
| step | lines | call |
|------|-------|------|
| build mcast addr (once, pre-loop) | 308–310 | `get_noc_multicast_addr<MCAST_NOC_INDEX>(...)`, `brisc_mcast_sem_addr = noc_addr | sem_addr` |
| set local sender flag VALID (pre-loop) | 316–317 | `noc_semaphore_set(mcast_semaphore_ptr, MCAST_VALID)` |
| read source DRAM→L1 | 342–343 | `noc_async_read` + `noc_async_read_barrier` |
| **pre-handshake: wait receivers ready** | 348–349 | `noc_semaphore_wait(receiver_ready_semaphore_ptr, num_mcast_dests)` → reset 0 |
| **issue DATA mcast** | 352–358 | `noc_async_write_multicast(k_write_ptr, mcast_dest_addr, size, num_dests, false, MCAST_NOC_INDEX)` |
| **issue SEM flag mcast** | 360–365 | `noc_semaphore_set_multicast(mcast_sem_addr, brisc_mcast_sem_addr, num_dests, false, noc)` |
| flush | 366 | `noc_async_writes_flushed(MCAST_NOC_INDEX)` |

RECEIVER half: cores wait on `mcast_semaphore` ==MCAST_VALID and signal `receiver_ready` via `unicast_atomic_inc_set_state` (lines 318–320, 491+).

## Forks
- **F1 = flush** (line 366 `noc_async_writes_flushed`).
- **F2 = flag** (MCAST_VALID level flag, set pre-loop at 317; receiver resets). Distinct from F2-counter.
- **F3 = EXCLUDE_SRC** (`loopback`/SRC_INCLUDE not set; sender separate from dest grid).
- **pre_handshake = PRESENT and load-bearing** (lines 348–349). Receivers atomic-inc a `receiver_ready` counter; sender waits == num_dests before each mcast. Dest L1 slot is reused across loop iters → the pre-handshake is what makes reuse safe. This is the **KNOB=pre_handshake "dest reused"** branch in the wild.

## HOLE flags
- Pre-handshake uses a **counter** (`receiver_ready`, atomic-inc up to num_dests) while the data-ready signal is a **flag** (MCAST_VALID) — same mixed-F2 pattern as coordinator_kernel. Pipe must support a counter for the ready-collect and a flag for the data-ready, simultaneously.
- mcast addr built once outside the loop, reused with `| k_write_ptr` per iter (line 351). Good Pipe pattern: build rectangle once, OR in the per-iter L1 offset.
