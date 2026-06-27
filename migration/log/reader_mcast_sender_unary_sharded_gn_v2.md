# reader_mcast_sender_unary_sharded_gn_v2.cpp — FAILED (multi-rect)

- Tier: 2 (refactor-low), worklist item 1.
- Role: sender (group_norm sharded, legacy / non-welford).
- Validation nodeid (gn_v2 legacy): `test_group_norm_with_block_sharded_v2_8x4_grid[specify_grid=True-legacy-...num_groups=32...]`.

## Verdict: API MISMATCH — multi-rect. NOT MIGRATED. Tree untouched.

## Why it does not fit the Pipe API
The sender mcasts the reduce result + flag to **up to THREE distinct NoC rectangles per logical
exchange**: a "mid" group (always), an optional "first" group, and an optional "last" group. Each
rectangle has its OWN corner coords AND its OWN num_dests:
- `num_mcast_cores_mid_group`   → rect {mcast_dest_noc_start/end_x/y}
- `num_mcast_cores_first_group` → rect {mcast_first_group_dest_noc_start/end_x/y}  (if has_mcast_first_group)
- `num_mcast_cores_last_group`  → rect {mcast_last_group_dest_noc_start/end_x/y}   (if has_mcast_last_group)

The data is written via three separate `noc.async_write_multicast(...)` calls and the flag via three
separate `reduce_sender_sem.set_multicast(...)` calls, followed by a single trailing
`noc.async_write_barrier()`.

`dataflow_kernel_lib::McastRect` is a SINGLE rectangle with a SINGLE `num_dests`. KNOWN LIMIT #1(a):
the Pipe cannot express a send to a LIST of rectangles. Confirmed in the normalization audit
(`migration_audit/normalization.md` line 27: "multi-rectangle (×3)"; headline blocker #5: "all
groupnorm senders mcast to up to 3 NoC rectangles ... Pipe.send() must accept a list of rectangles").

## Action
No edits made (kernel was untouched). Marked FAILED + quarantined. No commit, no revert needed.
