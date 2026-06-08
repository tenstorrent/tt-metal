# reader_interleaved.cpp (sdpa) — TIER 3.3 (refactor-high)

**Status: FAILED (left untouched, no commit)**
**Validation: n/a** (no edit; tree green)

## Op / dispatch
ttnn.transformer.scaled_dot_product_attention prefill (SDPAProgramFactory). KV chain forwarding,
non-causal only, `mcast_enabled` all-or-nothing.

## Handshake blocks (open-coded chain-link, K and V)
- **Receive (L393-400)** — `cb_reserve_back; noc_semaphore_set(receiver_sem, INVALID); noc_semaphore_inc(sender_sem_noc_addr,1); noc_semaphore_wait(receiver_sem, VALID); cb_push_back`.
- **Forward / chain-link send (L447-468)** — `noc_semaphore_wait(sender_sem, wait_count); noc_semaphore_set(sender_sem,0); noc_async_write_multicast(linked=true); noc_semaphore_set_multicast(valid_sem); noc_async_writes_flushed()`.

## Assessment — NO FIT
1. **Legacy raw free-function API, not the object API.** The block uses `noc_async_write_multicast`,
   `noc_semaphore_set_multicast`, `noc_semaphore_set/inc/wait`, `get_noc_addr`,
   `get_noc_multicast_addr`, and raw `volatile tt_l1_ptr uint32_t*` semaphore pointers. The Pipe is
   built strictly on `Noc`/`Semaphore<>`/`MulticastEndpoint`. There is no `Noc`/`Semaphore<>` in this
   kernel. A Pipe migration would require first porting the whole chain-forwarding block (address
   arithmetic `mcast_base_noc_addr | cb_k_start_address`, the semaphore pointers, num_dests/wait_count
   runtime args) to the object API — that is a rewrite, not a surgical replacement of the handshake
   block. (Matches the audit's "legacy raw API → port to Noc/Semaphore first" verdict for move/sort.)
2. **R6 ring / role-flip topology.** Each chain participant both RECEIVES K/V from the previous core
   and FORWARDS to the next (`should_receive`/`should_forward` on the same core, same iteration),
   threaded through the chain metadata and a `valid_semaphore` self-prime. This is exactly the
   rotating-sender / role-flip pattern the bake-off proved hangs as a single same-core Pipe, and
   "sdpa ring legs" is on the invocation's deferred/out-of-scope list. The Pipe's INV9 precondition
   (single sender per receiver) and `McastRect` single-rect model do not cover ring forwarding.
3. Even the isolated forward send (LINK=true linked data+flag pair + flush) maps shape-wise to a
   `Pipe<EXCLUDE_SRC, Flag, PRE_HANDSHAKE=true, LINK=true>::send()`, but the surrounding
   raw-address build and the ring receive on the same core make any partial non-surgical and unsafe.

Left RAW.
