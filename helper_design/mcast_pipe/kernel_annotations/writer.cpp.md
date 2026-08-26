# `writer.cpp` (Conv3D) — migrated annotation

Path: `ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/writer.cpp`

Role: hybrid sender/receiver/passive, selected by compile-time weight-share mode and per-core runtime
role. API-v11 `Mcast2D` owns only the rectangle multicast mode.

## Current multicast protocol

- `WeightMcastArgs` is chained after output, weight, and bias TensorAccessor compile-time arguments and
  owns runtime words `[19, 23)`. Operation parsing resumes through `next_runtime_args_offset()`.
- A fixed sender reads one weight block from DRAM into `cb_weight`, constructs the helper sender face,
  and calls guarded `send()` with equal source and destination L1 addresses. Because the sender is inside
  the rectangle and the addresses alias, API v11 infers EXCLUDE-source multicast; the DRAM-read local
  copy is retained for sender compute.
- An active receiver reserves the weight CB, calls `receive()`, and publishes the block to compute.
- A passive receiver constructs the same receiver face, calls `receive()` once per expected iteration,
  consumes no CB entries, performs the original final atomic barrier, and exits.
- The helper's pre-handshake Counter and level Flag preserve the raw ACK/wait/reset protocol. Default
  `SourceL1Guard` completes the remote send before the sender can reuse its single-block source buffer.

## Host invariants

- Every group strip is a dense logical rectangle with a fixed sender inside it.
- Each exact per-group `Mcast2D` is active and has the same compile-time block as the representative
  helper object appended to the writer descriptor.
- The existing sender and receiver semaphore descriptors retain their allocation order. Their IDs are
  adopted as helper consumer-ready and data-ready IDs respectively.
- The weight CB descriptor covers `CoreRangeSet(core_grid)`, guaranteeing valid destination L1 storage
  for active and passive rectangle members.
- All modes receive the helper's four runtime words, using an inactive helper block outside Mcast mode,
  so the writer ABI remains unconditional.

## Explicitly out of scope

The Chain path is peer-to-peer forwarding with per-hop unicast writes, barriers, and atomic signaling;
it is not multicast and remains raw. Disabled/local weight loading also remains unchanged. The helper
objects are constructed only inside the Mcast branches, so neither sibling path changes behavior.

## Validation state

Migrated at API v11 in `a290ce202811f6867a0c18e1ebcb19285369881c`. Fresh-JIT exact correctness,
12 focused shapes, full unit and nightly inventories, shared helper guards, and matched non-grouped and
grouped performance gates passed. The two measured device-kernel medians improved 0.815% and 0.298%.
