# reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp — TIER 3.4 (refactor-high)

**Status: FAILED (left untouched, no commit) — "R4 streaming deferred"**
**Validation: n/a** (no edit; tree green)

## Op / dispatch
ttnn.conv2d block-sharded activation reader (halo, 3x3 weights v2). Object API
(`Noc`/`Semaphore<>`/`MulticastEndpoint`). `act_mcast_sender_sem`, `act_mcast_receiver_sem`.

## Mcast / handshake blocks
- **The ONLY data mcast is the chunked streaming send** `mcast_block_chunked<...>(...)` (L281-286),
  which interleaves `src_cb_obj.wait_front(wait_tile_curr)` with per-burst `multicast_data` calls of
  ≤ NOC_MAX_BURST_SIZE on a NOT-yet-complete block (L102-124). This is precisely R4 (streaming
  chunked send), which the invocation marks DEFERRED RAW.
- Sender handshake around it (L276-314): `act_mcast_sender_sem.wait(num_dests + (recv?0:1)); set(0);
  act_mcast_receiver_sem.set(INVALID); <chunked send>; then flag broadcast`
  `act_mcast_receiver_sem.set(VALID); set_multicast<INCLUDE_SRC or EXCLUDE_SRC>(num_cores+1)` with a
  trailing `async_write_barrier` only on the INCLUDE_SRC branch.
- Receiver handshake (L315-329): `act_mcast_receiver_sem.set(INVALID); act_mcast_sender_sem.up(sender,1); act_mcast_receiver_sem.wait(VALID)`.

## Assessment — FAILED ("R4 streaming deferred")
Per invocation rule 4: migrate ONLY a fully-ready loopback/handshake broadcast; leave the chunked
streaming send (R4) RAW; "If the only mcast here is the chunked one, FAILED 'R4 streaming deferred'."
- There is **no fully-ready single-shot data broadcast** in this kernel — the sole data mcast is the
  chunked/streaming `mcast_block_chunked` (producer-overlapped, wait_front interleaved with bursts).
  The Pipe `send()` only handles a fully-ready block; it cannot wrap this.
- The flag broadcast (`act_mcast_receiver_sem.set(VALID)+set_multicast`) is structurally inseparable
  from the chunked data send it certifies: the comment at L288-290 documents that NO write barrier
  sits between the chunked data mcast and the flag mcast precisely so they ride the same NoC/VC
  (INV4) — extracting the flag into a `send_signal()` (which adds a `fence_`/flush) would break that
  tuned single-VC data→flag ordering, and the data it proves arrival of is the deferred chunked one.
- The receiver handshake (clear-before-ack `set(INVALID); up; wait`) likewise gates the deferred
  chunked send; migrating it alone (Pipe `receive()` clears after wait, H11) while the sender keeps
  raw chunked+flag would split the INV4 pair across sides and invert the reset ordering.

The kernel's loopback F3 tri-path (INCLUDE_SRC num_cores>0 / local-self-write num_cores==0 /
EXCLUDE_SRC non-receiver) is already factored into `multicast_data<>` and is itself part of the
chunked send. Nothing fully-ready remains to migrate. Left RAW.
