# writer_single_row_multi_core.cpp — DEFERRED (design-gap)

Tier 3. Status: deferred. No code change.

## Role
Sort worker writer — confirmation-emit only: after writing each processed pair it
`up()`s `cores_to_coordinator_sem` (atomic-barrier inc). This is the RETURN LEG of the
coordinator's confirmation counter, not a Pipe face (no mcast, no data-ready flag here).

## Why deferred
- Pure ack emit; there is no SenderPipe/ReceiverPipe channel to wrap on this kernel. The
  `up()` is the receiver-side ack that the helper's `ReceiverPipe::receive()` would issue
  internally — but the receive lives in `reader_single_row_multi_core.cpp`, and the GO/data
  channel lives there + the coordinator. This file only carries the counter increment.
- Bound to the coordinator/reader STAR, which defers (runtime recipient count + split
  mcast/ack count + runtime sem ids). No standalone migration.

Helper untouched. Lines removed: 0.
