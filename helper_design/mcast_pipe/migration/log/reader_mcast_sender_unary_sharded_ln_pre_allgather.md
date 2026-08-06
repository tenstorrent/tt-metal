# reader_mcast_sender_unary_sharded_ln_pre_allgather — API v10 migration

**Tier:** 3, `layernorm-sharded-pre-allgather`
**Status:** fully end-to-end migrated at API v10, commit `4acd98259b6`
**Prerequisite:** shared reader-builder split in `4ef7e9a57a6`

## Transform

The host now emits one opaque handshaked Flag channel through
`PreAllGatherMcast`: whole-grid reductions use `Mcast2D`; two-stage reductions
use one `Mcast1D` per row or column. The production kernel decodes the wire
with `McastArgs<0, 0>` and replaces the raw ready-set, consumer-count wait,
counter reset, and semaphore multicast with `send_signal()`.

Gather reads, CB ownership, the second-stage semaphore, and the final write
barrier remain operation-owned.

## Validation

- Separate prerequisite: host build passed; 126 pre-allgather, 136
  post-allgather, and 208 plain sharded cases passed unchanged.
- Production host build passed.
- Exact 8x4 BFLOAT8_B RMSNorm node passed under `--dev` from fresh isolated
  cache; sender ELF confirmed.
- Complete inventories: 126 pre-allgather, 136 post-allgather, 208 sharded.
- `McastHostFixture`: 28/28; `test_mcast_pipe.py`: 77/77.
- Exact-node profile: device-kernel durations 2,583, 2,564, 2,563, and 2,656
  ns; median 2,563.5 ns. Comparable pre-migration delta: N/A.
