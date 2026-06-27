# worker_writer.cpp — annotation (experimental/transformer/all_reduce_create_qkv_heads)

Path: `ttnn/cpp/ttnn/operations/experimental/transformer/all_reduce_create_qkv_heads/device/kernels/dataflow/worker_writer.cpp`
Role: writer (BRISC). MOSTLY OUT OF SCOPE — the data movement is **fabric** mcast, not local
NoC rectangle mcast. ONE in-scope fragment: a sem-only rectangle mcast (step 3).

## OUT OF SCOPE — fabric data mcast (lines 92-157)
- L92-95 `pkt_hdr_*->to_chip_multicast(MulticastRoutingCommandHeader{...})` — CROSS-CHIP fabric mcast.
- L107-134 `write_and_advance_local_read_address_for_fabric_write(...)` over FabricConnectionManager.
- L137-157 sem-inc via fabric (`to_noc_unicast_atomic_inc`, `send_payload_flush_blocking_from_address`).
- L164 `noc_semaphore_wait(out_ready_sem_bank_addr, out_ready_sem_wait_value)` — wait for all chips'
  fabric sem incs. This is a fabric all-reduce barrier, NOT a rectangle handshake.
These belong to the tt_fabric / ring-allreduce family. DEFER — not a local rectangle mcast Pipe.

## IN SCOPE (marginal) — sem-only rectangle mcast (lines 166-184)
- L55 `noc_semaphore_set(reduction_semaphore_send_addr_ptr, VALID)` — arm the flag (setup).
- L167-181 loop over `num_mcast_ranges`:
  - L169-174 `get_noc_multicast_addr(mcast_dest_noc_start_x[i], _start_y[i], _end_x[i], _end_y[i], reduction_semaphore_send_addr)` — rectangle.
  - L176-180 `noc_semaphore_set_multicast(reduction_semaphore_send_addr, recv_noc_addr, i==0 ? num_mcast_cores : 0, false)` — sem-flag mcast to the reduction-worker rectangle. linked=false.
- L184 `noc_semaphore_set(out_ready_sem_bank_addr, 0)` — reset the *fabric* sem (not this mcast's).
- L190 `noc_async_write_barrier()` — teardown drain.

Receiver = `reduction_receiver.cpp` L91 `noc_semaphore_wait(signal_semaphore_addr_ptr, VALID)` +
L92 reset-0. (reduction_receiver also has L171 a generic end barrier.)

## VARIANT TAGS (sem-only fragment)
- This is a DATA-LESS Pipe: `send()` carries only the 4-byte sem flag, no `noc_async_write_multicast`.
- **F1 = BARRIER** (L190, but it's teardown, not per-send).
- **F2 = FLAG** (VALID set L55, mcast L176, receiver wait+reset L91-92).
- **F3 = EXCLUDE_SRC** — the sender is an all-gather worker, receivers are separate reduction workers;
  no loopback.
- **KNOB pre_handshake = N/A** — no source-buffer reuse handshake; one-shot signal after a fabric barrier.
- Multi-range quirk: count `num_mcast_cores` charged only on i==0 (L179), later ranges pass dests=0 —
  a single logical broadcast split across several NoC rectangles. Pipe with multi-rectangle dest sets
  would need this "charge dests once" trick.

## HAZARD / INVARIANT mapping
- The sem mcast (L176) happens AFTER the fabric barrier (L164) — ordering across the two protocols is
  the fabric sem, not a NoC handshake. If folded into Pipe, the data-availability precondition is
  external (fabric), so Pipe::send(sem-only) must assume data already landed.

## VERDICT
DEFER-RAW for the data path (fabric, different family). The sem-only rectangle mcast (L166-184) is a
degenerate Pipe (no data) and could in principle use Pipe::send with a null data block, but the
multi-range "charge once" semantics and fabric coupling make it a poor first migration target.
