# writer_local_topk.cpp — FAILED (not rectangle-mcast)

- Tier: 2 (refactor-low), worklist item 5.
- Role: sender (ttnn.topk, TopKMultiCoreProgramFactory).
- Validation nodeid (topk W=8192): `tests/ttnn/unit_tests/operations/reduce/test_topk.py::test_topk[sub_core_grids=None-largest=True-sorted=True-N=1-C=1-H=32-W=8192-dim=3-k=50-BFLOAT16_B]`.

## Verdict: NOT A PIPE FIT — unicast scatter. NOT MIGRATED. Tree untouched.

## Why it does not fit the Pipe API
Census said "unicast scatter + up(counter)" — confirmed. Per height row this writer:
1. `receiver_sem.wait(VALID)` — WAITS the final core's go-flag (it is the *receiver* of this flag,
   not a broadcaster of it).
2. For each of Kt value tiles and Kt index tiles: `noc.async_write(...)` to ONE specific final core
   (`noc_final_x/y`) at a per-core offset address (`final_values_base = ... + start_wt * tile_bytes * Kt`).
   This is point-to-point UNICAST with a different landing offset per sender core — i.e. a SCATTER
   into the final core's aggregation buffer, NOT a rectangle multicast broadcast.
3. `noc.async_write_barrier()`.
4. `sender_sem.up(noc, noc_final_x, noc_final_y, Kt)` + `async_atomic_barrier()` — R->final COUNTER.
5. `receiver_sem.set(INVALID)`.

There is NO `async_write_multicast` / `set_multicast` anywhere — no rectangle, no broadcast.
KNOWN LIMIT #2: unicast-scatter (different data/addresses to one target, not a rectangle broadcast)
is not a Pipe fit. The only flag broadcast in this op is the final core's go-flag, which is RAISED by
the *final* core (reader_final_topk, the counterpart receiver-side kernel, already Tier-1 clean) and
merely WAITED here — there is no genuine rectangle flag broadcast in THIS kernel to migrate.

## Action
No edits made (kernel untouched). Marked FAILED + quarantined. No commit, no revert needed.
