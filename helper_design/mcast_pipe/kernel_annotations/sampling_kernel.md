# sampling_kernel.cpp (deepseek_v3_b1 micro_ops) — annotation

File: `models/demos/deepseek_v3_b1/micro_ops/sampling/kernels/sampling_kernel.cpp`
Role: SENDER+RECEIVER, NCRISC. The relevant block is the **single-device loop barrier** (lines 192–220), only in `!mesh_mode`. INTRA-CHIP. IN SCOPE.

## THE BLOCK — sem-only mcast loop barrier, lines 192–220
Final core releases the other cores between internal iterations.

SENDER half (is_final_core), lines 199–214:
| step | lines | call |
|------|-------|------|
| build mcast addr | 206–207 | `get_safe_multicast_noc_addr(start,end, 0)` (wraps `get_noc_multicast_addr` with NOC1 coord swap, lines 19–33) |
| OR sem addr into mcast addr | 208 | `mcast_sem_addr = mcast_noc_addr | local_ready_sem_addr` |
| set local flag | 210 | `noc_semaphore_set(local_ready_sem_ptr, 1)` |
| **mcast SEM flag** | 211 | `noc_semaphore_set_multicast(local_ready_sem_addr, mcast_sem_addr, num_dests)` |
| barrier | 212 | `noc_async_write_barrier()` |
| reset local flag | 213 | `noc_semaphore_set(local_ready_sem_ptr, 0)` |

RECEIVER half (non-final cores), lines 216–217:
`noc_semaphore_wait(local_ready_sem_ptr, 1)` → `noc_semaphore_set(local_ready_sem_ptr, 0)`.

## Forks
- **F1 = barrier** (line 212).
- **F2 = flag** (level flag 1; receiver waits ==1 then resets to 0). Lines 210/213/216/217.
- **F3 = EXCLUDE_SRC** (final core not in dest grid; `num_dests` is the others). No loopback.
- **pre_handshake = ABSENT** here (loop barrier, no receiver-ready collection).
- **No data block** — sem-only mcast. This is the "flag-only" degenerate of THE BLOCK.

## HOLE flags
- NOC coord swap is hand-rolled in `get_safe_multicast_noc_addr` (lines 20–33) — duplicate of the swap in mcast.hpp/sampling. The Pipe must own this once.
- Guarded by `num_dests > 0` (line 201) — Pipe needs a zero-dest no-op path.
