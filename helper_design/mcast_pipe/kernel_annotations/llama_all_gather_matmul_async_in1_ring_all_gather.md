# llama_all_gather_matmul_async / reader_bmm_tile_layout_in1_ring_all_gather.cpp — annotation (added by reconcile 2026-06-19)

File: `ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/reader_bmm_tile_layout_in1_ring_all_gather.cpp`
Family: ccl. Role: sender. Tag: **defer**.

The CCL twin of the census's matmul `reader_bmm_tile_layout_in1_ring_all_gather.cpp` (tagged `defer`,
P4 "flag-only collective barrier, no data half"). Same shape here: a **flag-only ring/collective
barrier** with no data-multicast half.

## BLOCK
| step | lines | call |
|------|-------|------|
| build mcast addr (signalling sem rect) | 54 | `get_noc_multicast_addr(...)` |
| wait on ring partner's sem value | 56 | `noc_semaphore_wait(pv_semaphore_ptr, target_sem_value)` |
| **mcast FLAG (signal next in ring)** | 58 | `noc_semaphore_set_multicast(pv_semaphore, signalling_semaphore_address, num_signalling_semaphores)` |

## Why DEFER (out of scope this round)
- **No data multicast half** — it's a pure collective/ring barrier (flag mcast + wait). The Pipe helper
  is for the *data-mcast → handshake → flush* block; a flag-only ring sync is the same scoped-out case
  as the existing P4 matmul ring_all_gather defer.
- **Ring topology** — the wait/signal is a ring step (partner-indexed), not a sender→receiver-rectangle
  broadcast. Scoped out alongside R4 streaming / CCL fabric.
- Carry as `deferred`; revisit only if a flag-only / ring arm is ever added to the helper.
