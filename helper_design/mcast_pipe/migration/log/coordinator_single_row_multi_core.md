# coordinator_single_row_multi_core.cpp — MIGRATED API v9

Tier 5 atomic unit: `sort-single-row-control`. Code: `7337302b564`.

## Migrated role

The coordinator is the sender face of a no-handshake Counter control Pipe over the dense full worker
grid. `McastArgs` decodes the host `Mcast2D` wire and `send_signal()` publishes row-start and
substage-start events. Counter staging preserves back-to-back events without requiring worker ACKs.

The reader-ready and writer-done counters remain explicit operation-owned semaphores. They are two
independent return channels with different phase counts, not one Pipe handshake. The former runtime
recipient-count and semaphore-ID blockers were stale ABI observations: this factory route activates
the full grid, helper semaphore ID 0 is constexpr, and `Mcast2D` derives the EXCLUDE-source fan-out.

Runtime args fell from 11 to 7. The host owns the multicast rectangle and sender-coordinate wire.

## Validation

- `./build_metal.sh`: passed.
- Exact `test_sort_long_tensor[shape=[1, 524288]-dim=-1-descending=False]` under `--dev` from a fresh
  isolated JIT cache: passed; coordinator, reader, and writer artifacts confirmed.
- `test_sort_multi_row_multi_core_no_deadlock`: 2 passed (`descending=False/True`, `Ht=2`).
- Complete `test_sort_long_tensor`: 7 passed.
- Complete `test_mcast_pipe.py`: 72 passed, including four control-only Counter cells.

Helper API remains v9.
