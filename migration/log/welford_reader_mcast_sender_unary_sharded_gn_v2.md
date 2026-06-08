# welford_reader_mcast_sender_unary_sharded_gn_v2.cpp — FAILED (multi-rect)

- Tier: 2 (refactor-low), worklist item 2.
- Role: sender (group_norm sharded welford).
- Validation nodeid (gn_v2 welford): `test_group_norm_with_block_sharded_v2_8x4_grid[specify_grid=True-welford-...num_groups=32...]`.

## Verdict: API MISMATCH — multi-rect. NOT MIGRATED. Tree untouched.

## Why it does not fit the Pipe API
Same structure as the non-welford gn_v2 sender. Per group (inner loop over num_groups) the sender
mcasts the combined `global_means_ptr` block (2 * single_tile_size_bytes) + the
`reduce_sender_sem` VALID flag to **up to THREE distinct rectangles** (mid / first / last), each with
its own corner coords and its own num_dests (`num_mcast_cores_mid_group`,
`num_mcast_cores_first_group`, `num_mcast_cores_last_group`), then a single trailing
`async_write_barrier()`.

KNOWN LIMIT #1(a): `McastRect` is single-rect with a single `num_dests` — cannot send to a list of
rectangles. Confirmed in `migration_audit/normalization.md` line 28 ("multi-rect; raw-L1 src;
per-group loop") and headline blocker #5.

## Action
No edits made. Marked FAILED + quarantined. No commit, no revert needed. The COUNTERPART receiver
(welford_reader_mcast_receiver_unary_sharded_gn_v2) WAS migrated (item 3) and validated green against
this untouched raw sender — the receiver's `receive()` (up + wait(VALID) + clear) is protocol-identical
to what this raw multi-rect sender drives.
