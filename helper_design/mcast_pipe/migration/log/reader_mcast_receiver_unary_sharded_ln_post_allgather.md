# Post-allgather sharded LayerNorm receiver — API-v11 migration

**Tier:** 2.10
**Status:** migrated at API v11
**Production commit:** `6cc49825476de78c4a86f3aada72c175ddffe095`
**Verified:** 2026-08-16 on single-chip Blackhole p100a

The receiver is migrated atomically with its sender twin. It consumes the helper-generated wire through
`McastArgs<0, 0>::receiver()` and `ReceiverPipe::receive()`, replacing the raw Flag wait/clear sequence.
The operation's circular-buffer reserve and publish remain outside the helper.

Validation shared with the sender:

- Release build and exact fresh-cache post-allgather JIT passed; both artifacts were present.
- `LN-POST-ALLGATHER` 136/136, `LN-PRE-ALLGATHER` 126/126, and `LN-SHARDED` 208/208 passed.
- Host fixture 34/34, helper device suite 80/80, and source audit before and after write-back 18/18 passed.
- Receiver LOC gate: 10 additions / 12 deletions.

No helper API change was needed. The operation-level non-`mcast_1d` coverage boundary is documented in
the sender log; its outside-sender receiver geometry is host-tested.
