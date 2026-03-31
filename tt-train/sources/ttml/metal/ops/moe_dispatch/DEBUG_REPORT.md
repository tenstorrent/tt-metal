# MoE Dispatch Kernel Debug Report

## Problem
The kernel hangs during the second `fabric_send_chip_unicast_noc_unicast_1d` call.

## DPRINT output (chip 0 = mesh coord 13)
```
SENDER: start, num_experts=8, E_local=1, me=13
SENDER: opening fabric barrier
SENDER: fabric open done
SENDER: expert=1 n_rows=1 dest=1
SENDER: remote row=0 dest=1 src_tile=4
SENDER: remote read start
SENDER: remote read done, fabric sending D_t=4 tiles
SENDER: remote fabric send done           ← first send OK (to device 1)
SENDER: expert=2 n_rows=2 dest=2
SENDER: remote row=0 dest=2 src_tile=4
SENDER: remote read start
SENDER: remote read done, fabric sending D_t=4 tiles
                                            ← HANGS HERE (second send to device 2)
```

No RECV or WRITER prints — receiver never gets past the semaphore barrier.

## Root cause analysis

### 1. Fabric connection setup mismatch

The kernel uses `fabric_connections[route]` where `route` is determined by
`get_route()` which returns a direction index:
- EAST=0, WEST=1, NORTH=2, SOUTH=3

For a [1,32] linear mesh:
- Sending from device 13 to device 1: route=WEST (dest < src on same row)
- Sending from device 13 to device 2: route=WEST (same direction)

Both go through `fabric_connections[WEST]`. The first send works, the second
hangs — likely because the fabric connection's internal state wasn't properly
handled between sends (the `wait_for_empty_write_slot` inside
`fabric_send_noc_unicast` may be spinning forever).

### 2. Program factory link_id ordering

The `all_to_all_dispatch` program factory iterates `neighbors` and uses an
incrementing `link_id`:
```cpp
uint32_t link_id = 0;
for (const auto& neighbor_coordinate : neighbors) {
    append_fabric_connection_rt_args(src, dst, link_id, ...);
    link_id++;
}
```

The kernel's `open_direction_connections_async` reads rt args sequentially for
each `directions[i]==true`. The order of `neighbors` from `get_neighbors` matches
the direction order (EAST first, then WEST, etc.).

My program factory does the same, so this should be correct. But `link_id` is
an ethernet channel index within the available links for that direction, NOT the
direction index. `append_fabric_connection_rt_args` internally determines the
direction from `(src, dst)` and validates that `link_id` is a valid forwarding
link for that direction.

### 3. The actual hang point

Inside `fabric_send_noc_unicast`:
```cpp
while (size_bytes > 0) {
    fabric_connection.wait_for_empty_write_slot();  ← likely hanging here
    ...
}
```

`wait_for_empty_write_slot` spins until the fabric EDM has consumed the previous
packet. If the EDM on the sender's ethernet core is stuck (e.g., the downstream
device isn't draining), this spins forever.

### 4. Why the first send works but the second doesn't

The first send goes to device 1 (from device 13). On a [1,32] linear mesh, the
fabric routes WEST through devices 12, 11, ..., 2, 1. Each intermediate device's
fabric EDM forwards the packet.

The second send goes to device 2 (also from device 13). Same WEST direction, same
fabric connection. The packet header specifies a `distance` of 11 hops. But the
fabric EDM may not have finished forwarding the first packet by the time we try
to send the second one through the same connection.

**The fabric connection is stateful** — it has a limited number of write slots
(typically 2 for double buffering). If we send too fast without the downstream
draining, we hang on `wait_for_empty_write_slot`.

In `all_to_all_dispatch`, the kernel sends one token at a time and only to ONE
destination per iteration of the inner loop. It doesn't burst-send to the same
direction without waiting for the receiver to process.

In our kernel, we send **D_t=4 tiles per row, one at a time** — 4 sequential
fabric sends for one row, then move to the next row for possibly the same dest
device. That's potentially many fabric sends without the receiver processing them.

### 5. The real problem: no flow control on the receiver side

`all_to_all_dispatch` uses a **global semaphore barrier** — the writer sends ALL
tokens, then increments the semaphore. The receiver waits for ALL senders to
finish before reading the output buffer.

But the fabric itself has limited buffering. If device 13 sends many tiles through
the fabric to device 2, the intermediate devices (12, 11, ..., 3) need to forward
them. If device 2's output buffer isn't being read (because the receiver is
waiting on the semaphore), the data still lands in DRAM on device 2 via the
fabric — it doesn't need the receiver kernel to accept it.

So the hang is NOT because the receiver isn't draining. The data goes directly to
DRAM. The hang is in the fabric connection itself.

### 6. Possible fix: send full rows as one packet, not tile-by-tile

Our kernel sends each [32,32] tile as a separate fabric packet (4 sends per row).
`all_to_all_dispatch` sends each token as one packet (aligned page size).

If the fabric max packet size is smaller than our row_bytes, we need multiple
packets per row. But if we align our page size correctly and send fewer larger
packets, the fabric may handle it better.

Also: `fabric_send_chip_unicast_noc_unicast_1d` internally loops over the payload
in chunks of `FabricMaxPacketSzBytes`. We're passing `(int)tile_bytes` as the
size, which is 2048 bytes — well within the max packet size. But we call it D_t=4
times per row, and then for the next row — that's 8 sequential fabric calls for
2 rows.

The fix might be to send the entire row (row_bytes) as one call instead of
tile-by-tile, since `fabric_send_noc_unicast` handles chunking internally.

## Action items

1. **Send rows as single call**, not tile-by-tile:
   ```cpp
   fabric_send_chip_unicast_noc_unicast_1d<...>(
       output_acc, fabric_connections, unicast_header,
       dest_device, l1, dst_tile_idx,
       (int)row_bytes, alignment);  // entire row, not tile_bytes
   ```
   But this requires `output_acc` to handle multi-tile payloads correctly.

2. **Match all_to_all_dispatch pattern exactly**: use the same page-by-page send
   pattern with proper alignment, matching the exact calling convention of
   `fabric_send_noc_unicast`.

3. **Verify fabric connection is not reused too aggressively**: add a barrier
   between consecutive sends to the same destination.
