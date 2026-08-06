# reader_mcast_receiver_unary_sharded_ln_pre_allgather — API v10 migration

**Tier:** 3, `layernorm-sharded-pre-allgather`
**Status:** fully end-to-end migrated at API v10, commit `4acd98259b6`
**Prerequisite:** shared reader-builder split in `4ef7e9a57a6`

## Transform

The host prepends the same opaque handshaked Flag channel used by the sender.
Whole-grid receiver cores call `receive_signal()`. In two-stage mode this
kernel also runs on the additional row/column coordinators: those line leaders
call `send_signal()`, while the other line members call `receive_signal()`.
This preserves the original multi-coordinator topology without duplicating a
raw wire.

Gather reads, CB ownership, `reduce_second_stage_semaphore_id`, and the final
atomic barrier remain operation-owned.

## Validation

- Exact 8x4 BFLOAT8_B RMSNorm node passed under `--dev` from fresh isolated
  cache; both receiver variant ELFs confirmed.
- Complete inventories: 126 pre-allgather, 136 post-allgather, 208 sharded.
- Offset whole-grid and row/column two-stage host geometry passed inside
  `McastHostFixture` 28/28; `test_mcast_pipe.py` passed 77/77.
- Production host build passed.
- Exact-node profile: device-kernel durations 2,583, 2,564, 2,563, and 2,656
  ns; median 2,563.5 ns. Comparable pre-migration delta: N/A.
