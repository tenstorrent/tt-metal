# Post-allgather sharded LayerNorm sender — API-v11 migration

**Tier:** 2.10
**Status:** migrated at API v11
**Production commit:** `6cc49825476de78c4a86f3aada72c175ddffe095`
**Verified:** 2026-08-16 on single-chip Blackhole p100a at 800 MHz

## Protocol mapping

- `mcast_1d` uses one dense `Mcast2D`; the sender is inside the rectangle and the helper's inferred
  loopback preserves the raw `MCAST_INCL_SRC` copy from CB21 to CB15.
- Non-`mcast_1d` uses one `Mcast2D` per row or column. The sender is outside that channel's receiver
  rectangle, so the operation performs its local CB21-to-CB15 asynchronous write and the helper sends
  only to the remote receivers. The local write is completed before the CB is published.
- Post-allgather remains no-handshake. The existing semaphore IDs are adopted, and helper compile-time
  and runtime blocks replace the raw geometry and semaphore words without changing unrelated arguments.
- The host constructs channels over the dense destination bounding boxes. All landed cells therefore
  own the required circular buffer and semaphore state, including inactive cells in a ragged logical grid.
- No helper implementation or public API changed. API expansion: **NO**.

## Validation

- `./build_metal.sh`: passed.
- Exact post-allgather LayerNorm under `--dev --no-precompile` from a fresh isolated cache: passed with
  0/47 cache hits; sender and receiver artifacts confirmed.
- `LN-POST-ALLGATHER`: 136/136 passed.
- ABI guards: `LN-PRE-ALLGATHER` 126/126; `LN-SHARDED` 208/208.
- `McastHostFixture.*`: 34/34, including offset dense and per-row outside-sender configurations.
- `test_mcast_pipe.py`: 80/80. Source audit before and after ledger promotion: 18/18.
- Per-file LOC gate: sender 26 additions / 45 deletions; every other production file in the atomic unit
  also independently has fewer additions than deletions.

## Matched performance

Each result is the median of three independent 25-operation Tracy sessions after discarding the first
five samples per session:

- LayerNorm: raw 4009 ns, migrated 3880 ns, **-3.2%**.
- RMSNorm: raw 4020 ns, migrated 3617.5 ns, **-10.0%**.

## Coverage boundary

The mapped operation inventory currently dispatches only `mcast_1d`. A direct non-`mcast_1d` operation
probe cannot provide valid per-line inputs: the public operation validates the post-allgather stats
tensor as sharded on exactly one core, while the non-1D topology has multiple sender lines. Only the
first line consequently owns stats. This is a pre-existing operation-contract limitation, not a helper
API gap; the outside-sender wire and argument geometry are covered by the host fixture.

## Historical context

The v8 migration was rolled back during the v9 review because the then-proposed mapping expected helper
loopback for an out-of-rectangle sender. Tier 2.10 follows the rollout plan's explicit remediation:
retain the existing helper for the remote rectangle and make the sender's local copy operation-owned.
