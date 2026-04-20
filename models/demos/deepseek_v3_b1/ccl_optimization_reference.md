# DeepSeek CCL Optimization Reference

## Purpose

This note captures the all-reduce/fabric-side optimizations that were developed on
`aagarwal/channel-trimming-test` so they can be reapplied to other DeepSeek V3 B1 CCLs
without replaying the full commit history.

The intent is to document the semantic optimizations, their contracts, and the
conditions under which they are worth reusing. This is not meant to preserve the
temporary micro-profiling scaffolding used to discover them.

## Scope

This reference covers the writer/fabric-path optimizations that landed across the
following files:

- `models/demos/deepseek_v3_b1/unified_kernels/all_reduce.hpp`
- `tt_metal/fabric/hw/inc/tt_fabric_api.h`
- `tt_metal/fabric/fabric_edm_packet_header.hpp`
- `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp`

It intentionally does not try to summarize unrelated functional drift in
`all_reduce.hpp`.

## Source Branch And Commits

Semantic optimization commits on `aagarwal/channel-trimming-test`:

- `e318813d2f8` `optimize set fabric route`
- `b5f63b44b76` `optimize pkt header updates`
- `dd824e0ca5d` `free slots caching optimization`
- `dcf2308c6c8` `initial stateful send opt`
- `e274743d6d9` `switch to posted from non-posted`
- `f2142c83744` `add multi-header support`
- `4df50f30671` `move code around + remove atomic barrier`
- `ec102723651` `update header count logic`

Profiling-only commits used during optimization, but not required for semantic ports:

- `bfaeb490749` `profiling changes`
- `0ec31fdf01c` `remove micro-profiling zones`

## High-Level Summary

The all-reduce writer optimization work followed this progression:

1. Remove fixed setup overhead in fabric route programming.
2. Stop rebuilding packet headers when only a few fields change per packet.
3. Amortize downstream free-slot queries across bursts.
4. Reuse NOC transport state for payload/header sends to the downstream EDM.
5. Relax the hot path from non-posted to posted writes where completion acks were not needed.
6. Reduce flush frequency by rotating between pre-materialized headers.
7. Remove an unnecessary immediate atomic barrier from the sender-side local-ready notification.

The common pattern is:

- precompute as much per-connection state as possible
- mutate only the fields that vary per packet
- amortize or eliminate control-path polling
- match synchronization strength to the actual protocol requirement

## Optimization 1: Single-Hop Fabric Route Fast Path

### Motivation

The generic `fabric_set_unicast_route(...)` helper was paying for general 2D route
decode/setup even though the all-reduce writer was doing a simple neighbor exchange
to a final destination that was exactly one `E/W/N/S` fabric hop away.

### What Changed

- `tt_metal/fabric/hw/inc/tt_fabric_api.h`
  - added `single_hop_route_cmd_by_direction`
  - added `fabric_set_single_hop_unicast_route_from_direction(...)`
  - added `fabric_set_single_hop_unicast_route(...)`
- `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp`
  - the worker connection now stores `edm_direction`
  - added `get_connection_direction()`
- `models/demos/deepseek_v3_b1/unified_kernels/all_reduce.hpp`
  - switched the writer setup path to use the direction-based fast path

### Contract

Use this fast path only when all of the following are true:

- the packet is worker-originated
- the destination is the final destination
- the path is exactly one physical `E/W/N/S` hop
- the path does not cross `Z` / inter-mesh routing boundaries

### Why It Helps

This removes the generic route decode/fill work from the writer setup path and
replaces it with a direct route-buffer initialization for the single-hop case.

### Porting Notes

Reuse this only for CCLs that have the same one-hop final-destination contract.
If a CCL can route through `Z`, multi-hop, or recompute boundaries, keep the generic
helper.

## Optimization 2: Partial Packet Header Mutation

### Motivation

Once the route setup cost was reduced, the next fixed cost was repeated packet-header
materialization even though only a small subset of header fields changed from packet to
packet.

### What Changed

- `tt_metal/fabric/fabric_edm_packet_header.hpp`
  - added `set_payload_size_bytes(...)`
  - added `set_fused_unicast_write_atomic_inc_write_noc_address(...)`
- `models/demos/deepseek_v3_b1/unified_kernels/all_reduce.hpp`
  - prebuilds the fused unicast-write-atomic-inc header once
  - updates only the destination NOC address per regular packet
  - updates payload size only for the tail packet when needed

### Why It Helps

The writer no longer rebuilds the entire fused header for every packet. It keeps the
invariant fields fixed and mutates only:

- the write destination address
- the payload size for the final short packet

### Porting Notes

This pattern is broadly reusable across CCLs whenever:

- the packet format is stable
- only the payload address and optional tail size vary per send

If a CCL changes many command fields per packet, the benefit will be smaller.

## Optimization 3: Free-Slot Query Amortization

### Motivation

The writer was repeatedly polling downstream free-slot availability one packet at a
time, even in bursty steady-state loops where multiple slots were already available.

### What Changed

- `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp`
  - added `get_num_free_write_slots()`
  - redefined `edm_has_space_for_packet<num_slots>()` in terms of the bulk count
- `models/demos/deepseek_v3_b1/unified_kernels/all_reduce.hpp`
  - introduced `cached_free_write_slots`
  - refills the cached count only when it reaches zero
  - drains multiple packets from one refill

### Why It Helps

This converts a repeated control-path query into an amortized burst-level query.

### Porting Notes

This is useful for any writer loop that:

- repeatedly sends to the same downstream connection
- can consume multiple free slots back-to-back once they are available

## Optimization 4: Stateful Worker-to-Router Transport

### Motivation

After trimming setup and header-update costs, the next hot-path redundancy was the
transport itself: payload and header writes kept reprogramming the same remote EDM core
state on every packet.

### What Changed

- `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp`
  - added `setup_stateful_send_cmd_bufs<posted>()`
  - added `send_current_slot_stateful_non_blocking<posted>()`
  - added private helpers:
    - `current_buffer_slot_l1_addr()`
    - `issue_payload_to_current_slot_stateful<posted>()`
    - `issue_header_to_current_slot_stateful<posted>()`
  - final implementation uses:
    - `ncrisc_noc_write_set_state(...)`
    - `ncrisc_noc_write_with_state(...)`
- `models/demos/deepseek_v3_b1/unified_kernels/all_reduce.hpp`
  - calls `connection.setup_stateful_send_cmd_bufs(...)` once after `open_finish()`
  - uses the combined adapter helper for each payload/header pair

### Why It Helps

The adapter pins the repeated transport state into command buffers once, then reuses
that state for:

- payload writes into the current downstream slot
- header writes into the same downstream slot
- downstream credit updates

This avoids reprogramming the same remote EDM/core state on every packet.

### Porting Notes

This pattern is a good fit when a CCL:

- repeatedly sends payload/header pairs to the same downstream EDM core
- has per-packet dynamic local destination addresses within a stable remote target
- can keep unrelated writes off the aliased command buffer while the stateful loop is active

Note the `DM_DYNAMIC_NOC` caveat in the adapter comments: generic worker-side writes on
the same RISC should not disturb the stateful send loop.

## Optimization 5: Posted Transport Writes

### Motivation

The all-reduce writer hot path needed departure ordering, but did not need an explicit
completion acknowledgement back to the sender for every packet write.

### What Changed

- `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp`
  - templated stateful send helpers on `posted`
  - kept the credit inline write explicitly non-posted
- `models/demos/deepseek_v3_b1/unified_kernels/all_reduce.hpp`
  - enabled `use_posted_transport_writes = true`
  - used the posted stateful path for payload/header transport
  - switched the send-loop drain to `noc_async_posted_writes_flushed()`
  - added a final posted drain before teardown

### Why It Helps

Posted writes avoid using a stronger completion model than the hot path actually
requires. The win is not just issue-time overhead; it is also a better match to the
sender's real ordering requirement.

### Important Caveat

If a path moves to posted transport, its drain logic must also move to posted-aware
flushes. Do not keep using non-posted-only drain helpers by mistake.

### Porting Notes

Reuse this only when the CCL hot path needs:

- departure ordering
- visibility ordering at the downstream EDM/router

but does not need:

- per-packet completion acknowledgement before continuing

Keep the downstream credit update non-posted unless there is a separate reason to
relax that path.

## Optimization 6: Multi-Header Ring To Reduce Flush Frequency

### Motivation

With a single reusable header, the writer had to flush aggressively because the next
packet needed to overwrite the same header storage.

### What Changed

- `models/demos/deepseek_v3_b1/unified_kernels/all_reduce.hpp`
  - introduced a small ring of pre-materialized headers
  - allocates headers once up front from `PacketHeaderPool`
  - fully initializes route and command fields once per header
  - rotates `header_idx` as packets are sent
  - flushes only when the header ring wraps

The final logic uses:

- `max_header_ring_size = 2`
- `packets_to_send_for_writer()`
- `header_ring_size = packets_to_send <= 1 ? 1 : 2`

### Why It Helps

The header no longer becomes a per-packet serialization point. With two headers, the
writer can overlap one in-flight header with preparation of the next packet and flush
only when it is about to reuse a header.

### Why The Ring Size Was Capped At Two

Profiling on the all-reduce writer showed that:

- one header is best when the writer only sends one packet
- two headers are the sweet spot for multi-packet writers
- beyond two, the extra setup cost outweighed the flush savings

### Porting Notes

When reusing this pattern in another CCL:

- start with `1` vs `2`, not a large header pool
- base the ring size on packets sent by the current writer, not total packets in the op
- materialize route/command state once per header outside the hot loop

## Optimization 7: Remove Immediate Atomic Barrier After Local-Ready

### Motivation

The sender-side local-ready path performed:

```cpp
noc_semaphore_inc(local_ready_noc_addr, 1);
noc_async_atomic_barrier();
```

The immediate atomic barrier added a large fixed cost even though the sender did not
have an immediate dependency on the completion acknowledgement of that semaphore write.

### What Changed

- `models/demos/deepseek_v3_b1/unified_kernels/all_reduce.hpp`
  - removed the immediate `noc_async_atomic_barrier()` after `noc_semaphore_inc(...)`

### Why It Was Safe For This Path

In the all-reduce local-ready handshake:

- the sender only needs to notify the receiver that local data is available
- the receiver already waits on the semaphore before it NOC-reads the sender data
- the sender does not need to know that the increment has been acknowledged before it
  continues with the rest of the writer work
- end-of-kernel barriers still provide the final accounting/cleanup guarantee

### When Not To Reuse This Blindly

Do not remove the immediate atomic barrier if the sender:

- immediately reuses or overwrites the same atomic target
- requires acknowledgement of the atomic completion before proceeding
- depends on the completion for correctness rather than one-way notification

## Profiling-Only Work

The optimization branch temporarily added micro-profiling zones to isolate:

- writer setup
- payload/header setup
- transport
- flush cost
- local-ready cost

That instrumentation was intentionally removed in the final semantic state. When
porting the optimizations to other CCLs:

- do not treat the profiling zones as part of the optimization
- keep top-level profiling and micro-profiling as separate modes

## Reuse Checklist For Other DeepSeek CCLs

Before porting these ideas into another CCL, check the following:

1. **Topology contract**
   - Is the downstream destination a one-hop final destination?
   - If yes, consider the single-hop route fast path.

2. **Header variability**
   - Are only the payload destination and optional tail size changing per packet?
   - If yes, pre-materialize the header and mutate only those fields.

3. **Send burst structure**
   - Can the writer consume multiple downstream slots once they are available?
   - If yes, cache free-slot counts instead of polling one packet at a time.

4. **Transport repetition**
   - Are payload and header repeatedly targeting the same downstream EDM core?
   - If yes, stateful send command-buffer setup is a good candidate.

5. **Synchronization strength**
   - Does the hot path need completion acknowledgement, or only ordered departure?
   - If only departure ordering is needed, consider posted transport writes.

6. **Flush pressure**
   - Is a single reusable header forcing a flush every packet?
   - If yes, try a small header ring, usually `1` or `2`.

7. **Semaphore semantics**
   - Is a sender-side atomic barrier really needed immediately after notification?
   - If the notification is one-way, the barrier may be removable.

8. **Measurement mode**
   - Keep shipping semantics and micro-profiling as separate configurations.

## Practical Porting Order

When propagating this work to another DeepSeek CCL, the recommended order is:

1. route fast path
2. partial header mutation
3. free-slot caching
4. stateful transport
5. posted transport writes
6. multi-header ring
7. local-ready barrier audit

This order mirrors the dependency chain from the original all-reduce work: first
remove fixed setup cost, then remove per-packet header cost, then trim transport and
flush overheads.
