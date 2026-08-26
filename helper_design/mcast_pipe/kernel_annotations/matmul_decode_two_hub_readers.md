# Matmul Decode two-hub readers — annotation

Added by reconciliation on 2026-08-16.

Files:

- `reader_full_width_sharded.cpp`
- `reader_partial_width_sharded.cpp`

Both live under
`ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/` and are byte-identical.
They bind independently through `full_width_sharded_program_factory.cpp` and
`partial_width_sharded_program_factory.cpp`.

## Required behavior

Every input-A shard sender unicasts its shard into one of two hub cores. Each hub waits for its assigned
shards, then multicasts its disjoint byte region of the shared `full_in0_cb` to the same compute rectangle,
including itself. After its data barrier, each hub increments the same `done_sem` on every receiver. Every
core waits for the count to reach two before publishing the complete CB.

- Destination set: full compute rectangle for both hubs.
- Sender membership: both hubs are also receivers; ordinary shard senders may overlap the compute set.
- Self-delivery: INCLUDE_SRC for each hub's data region.
- ACK population: full receiver count for each data multicast.
- Source lifetime: protected by an ACK barrier before the done increment.
- Completion: monotone Counter; exactly two independent hub increments are required.
- CB ownership: both producers write disjoint regions into one destination CB; publication occurs once,
  after both regions complete.

## Verdict

Role: **hybrid**. Intrinsic tag: **refactor-high**. Current rollout status: **deferred pending Tier-3
prototype**, not a settled capability verdict.

The helper documents a single-sender-per-receiver invariant. Unit 18 must first prototype composing two
existing no-pre-handshake Counter pipes and prove there is no premature CB publication or semaphore race.
Only a failed composition plus an independent production adopter could justify considering a generalized
multi-producer abstraction.
