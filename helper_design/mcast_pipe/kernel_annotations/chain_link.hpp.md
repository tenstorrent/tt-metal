# chain_link.hpp — annotation (transformer/sdpa)

Path: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chain_link.hpp`
Role: **shared header** that ALREADY abstracts the Pipe block. This is the reference
two-sided helper (`receive()` = Pipe::receive, `forward()` mcast branch = Pipe::send).
Consumers: only `ring_joint_reader.cpp` (call sites lines 138,162,399-417,475-492).

This file is the existing `ChainLink<mcast_enabled, is_head_level>` class. It is the
template the tune-helper `Pipe` should converge to / replace.

## THE BLOCK (mcast branch)

### Sender half — `forward(cb_addr, num_tiles, tile_bytes)`, lines 226-242 (mcast: 227-233)
- L228 `noc_semaphore_wait(sender_sem_ptr_, sender_wait_count_)` — handshake: wait until
  all `mcast_num_dests_` receivers signalled ready (KNOB pre_handshake = dest reused, sender
  blocks for receivers before reusing source buffer).
- L229 `noc_semaphore_set(sender_sem_ptr_, 0)` — reset counter (F2 = **flag/level reset**, not monotone).
- L230-231 `mcast_addr = mcast_base_noc_addr_ | cb_addr; noc_async_write_multicast(cb_addr, mcast_addr, num_tiles*tile_bytes, mcast_num_dests_, true)` — rectangle data mcast, **linked=true**.
- L232 `noc_semaphore_set_multicast(valid_sem_addr_, mcast_sem_noc_addr_, mcast_num_dests_)` — companion sem-flag mcast (the linked partner of the data write).
- L233 `noc_async_writes_flushed()` — **F1 = flush** (not barrier).

### Receiver half — `receive()`, lines 209-213
- L210 `noc_semaphore_set(receiver_sem_ptr_, INVALID)` — arm own valid-flag to INVALID before requesting.
- L211 `noc_semaphore_inc(sender_sem_noc_addr_, 1)` — signal sender "I am ready" (handshake up-count).
- L212 `noc_semaphore_wait(receiver_sem_ptr_, VALID)` — wait for the mcast sem-flag to flip VALID.

## Construction / addressing (lines 145-154)
- L147-148 `get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, 0)` — rectangle base, addr OR-ed at use site (L230). Injector only.
- L149 `mcast_sem_noc_addr_ = base | receiver_sem_addr` — sem rectangle addr.
- L140-143 valid-sem init to VALID (skip if addr 0).

## VARIANT TAGS
- **F1 = FLUSH** (`noc_async_writes_flushed`, L233). No barrier.
- **F2 = FLAG** (level VALID/INVALID + reset-to-0; sender_wait_count = mcast_num_dests). NOT monotone counter.
- **F3 = loopback**: EXCLUDE_SRC style. The injector does NOT mcast to itself — `should_receive`
  returns false for injector (L164) and self-fill is implicit (injector reads its own data into the CB
  via the normal DRAM read path, never receives). The rectangle is the receiver set only.
- **KNOB pre_handshake = YES (dest reused)**: sender waits on `sender_sem_ptr_` BEFORE writing
  (L228), because the source CB slot is the same one receivers read from on the prior iter.

## HAZARD / INVARIANT mapping
- **Linked-write companion ordering**: L230 (data, linked=true) MUST be immediately followed by
  L232 (sem mcast = companion). The class docstring of the open-coded twin (reader_interleaved
  L443-446) states: any `noc_async_read_barrier()` between them DEADLOCKS. The header preserves
  this by doing nothing between L231 and L232. INVARIANT: data-mcast + companion-sem-mcast are
  an atomic pair under linked=true.
- **Counter reset race (F2)**: L229 sets sender_sem to 0 after the wait; relies on receivers not
  incrementing again until next iter's `receive()`. Safe because forward() blocks future receivers
  via the valid-flag.
- **Unicast branch (L234-241) is OUT OF SCOPE** (point-to-point, no rectangle). It is the F3=INCLUDE
  alternative topology but not a rectangle mcast.

## HOLES
- HOLE: `forward()` mcast path does not issue a final barrier; correctness relies on caller-side
  ordering (compute cb_push_back after). Pipe must document the no-barrier contract or absorb it.
- HOLE: valid_sem_addr_ == 0 sentinel (L140) is an implicit "no-op" mode — Pipe needs an explicit
  disabled state or this becomes a silent footgun.
