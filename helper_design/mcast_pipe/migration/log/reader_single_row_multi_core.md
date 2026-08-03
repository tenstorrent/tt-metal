# reader_single_row_multi_core.cpp — MIGRATED API v9

Tier 5 atomic unit: `sort-single-row-control`. Code: `7337302b564`.

## Migrated role

The reader is the receiver face of the coordinator's no-handshake Counter control Pipe.
`McastArgs` decodes the shared `Mcast2D` wire and `receive_signal()` replaces the old inverted level
doorbell. The reader's row-ready `Semaphore::up` remains explicit operation protocol and uses the
helper wire's sender-coordinate escape hatch.

The helper owns semaphore ID 0; ready ID 1 is constexpr and op-owned. The old runtime semaphore IDs
and level reset are removed. The six-word runtime block is retained but now consists of two buffer
words plus four helper sender-coordinate words.

## Validation

- Host build passed.
- Exact fresh-cache `--dev` long-tensor node passed with this reader JIT artifact confirmed.
- Ht=2 deadlock regression: 2 passed.
- Full long-tensor inventory: 7 passed.
- Helper suite: 72 passed.

Helper API remains v9.
