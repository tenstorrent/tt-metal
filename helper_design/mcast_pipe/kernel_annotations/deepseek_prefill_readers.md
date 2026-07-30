# deepseek_prefill dispatch/combine readers — annotation

Files (fabric CCL ops; the local rectangle mcast leg is IN SCOPE):
- `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/reader_dispatch.cpp`
- `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/reader_combine.cpp`

Both have only 3 fabric refs (the cross-chip leg is elsewhere); the mcast here is a **sender→idle-cores intra-chip rectangle block**.

## reader_dispatch.cpp — BLOCK lines 164–268 (IS_TILE_LAYOUT)
SENDER:
| step | lines | call |
|------|-------|------|
| build 2 mcast addrs (value sem, ready sem) over idle-core bbox | 169–172 | `get_noc_multicast_addr(...)` ×2 |
| set local value | 255 | `*addr_value_ptr = untilize_base` |
| **mcast DATA (addr value, 4B)** | 256 | `noc_async_write_multicast(addr_value_sem_l1_offset, ..., sizeof(uint32_t), num_idle_cores)` |
| set 2nd local value | 261 | `*mbox_scratch_addr_ptr = rt_scratch_base` |
| build 2nd mcast addr | 262–263 | `get_noc_multicast_addr(...)` |
| **mcast DATA (mbox scratch, 4B)** | 264–265 | `noc_async_write_multicast(...)` |
| **barrier** | 266 | `noc_async_write_barrier()` |
| **mcast SEM (counter inc)** | 267 | `noc_semaphore_inc_multicast(mcast_addr_ready_noc_addr, 1, num_idle_cores)` |
| atomic barrier | 268 | `noc_async_atomic_barrier()` |

## reader_combine.cpp — BLOCK lines 166–246
Same shape: build mcast addrs (166/168) → `noc_async_write_multicast` data (241) → `noc_semaphore_inc_multicast(..., 1, num_idle_cores_group)` (246).

## Forks
- **F1 = barrier** (dispatch 266 `noc_async_write_barrier` + 268 `noc_async_atomic_barrier`).
- **F2 = COUNTER** — uses `noc_semaphore_inc_multicast` (NEW spelling vs `set_multicast`). Monotone counter; receivers `noc_semaphore_wait_min`. This is the clean F2-counter exemplar in the group (contrast the flag-based mcast.hpp/sampling).
- **F3 = EXCLUDE_SRC** (num_idle_cores; sender is the dispatch core, not idle).
- **pre_handshake = ABSENT** on send; a separate **return handshake** follows (idle cores NOC-write their mailbox addr into sender's scratch, then sender waits `mbox_ready == num_idle_cores`, lines 270–274) — a *post*-handshake, the inverse direction.

## HOLE flags
- `noc_semaphore_inc_multicast` is a distinct primitive from `noc_semaphore_set_multicast` — Pipe's F2 must offer both set(flag) and inc(counter) mcast.
- Multiple small 4-byte data mcasts back-to-back before one barrier — Pipe should batch barrier across multiple data sends.
- `noc_async_atomic_barrier` (not just write barrier) needed because the handshake is an atomic inc — separate flush primitive.
