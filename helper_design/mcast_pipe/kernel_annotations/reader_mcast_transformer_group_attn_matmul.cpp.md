# `reader_mcast_transformer_group_attn_matmul.cpp` — migrated annotation

Role: hybrid sender or receiver per `tile_row_id`; the first 32 logical cores rotate through the sender
role. The implementation uses API-v11 `Mcast2D` faces.

## Current protocol

- The host assigns helper semaphore IDs as compile-time arguments and passes the fixed dense receiver
  rectangle plus per-core sender coordinates through `McastArgs` runtime arguments.
- Runtime argument 20 remains operation-owned because each core has a divergent ACK count.
- The sender waits for the current round's receiver ACKs, resets the Counter, fills the source buffer,
  and calls `send()`. The receiver resets its Flag, ACKs the round's sender, and calls `receive()`.
- Both faces index the same round on every iteration: sender selection calls
  `McastArgs::sender_x/y(tile_row_id)` inside the tile-row loop, while
  `ReceiverPipe::receive(tile_row_id)` indexes the matching stored coordinate pair. No sender is
  latched from round zero.
- Sharded `CB2 -> CB1` sends include the source only when the sender is inside the receiver rectangle.
  Same-buffer `CB1 -> CB1` sends and outside-rectangle senders exclude the source.
- Send-only cores retain the existing immediate `CB1` push/pop behavior.

## Barrier proof

The raw kernel used a full write barrier after every flag multicast. It was redundant: API v11 flushes
the remote Flag write before returning from `send()` and barriers any local loopback write before the
sender resets helper state. A single final `noc.async_write_barrier()` preserves the only remaining
completion requirement—the last remote send before kernel exit.

The cross-round interlock is receiver-driven: `receive(N)` must observe and clear round N's Flag before
returning, and only the subsequent `receive(N+1)` issues that receiver's ACK to the next sender. The next
sender's Counter wait therefore cannot complete until every participating receiver has consumed the
prior Flag. Combined with the linked data+Flag chain and the sender's mode-appropriate fence before local
Flag reset, no later round can overtake an undelivered prior publication.

## Validation state

Migrated at API v11 in `6e8eb7638855ffc03948b34236b9219102215b49`. Exact fresh-JIT and complete
group-attention correctness passed. Matched 800 MHz q16/q48 performance improved 32.20% and 27.31%.
