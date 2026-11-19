# Balance heads while retaining partial Q/K groups

This independent candidate uses the passing packed prepared-Q/K helper, without
the unmeasured persistence fusion. The incumbent partitions B32's 512 complete
Q/K groups across 110 cores, leaving a 15-head critical-core assignment despite
only 13.96 heads per core on average.

The candidate partitions individual contiguous heads, so each core gets 13 or
14 heads. Compute receives the global head start as a second runtime argument.
It prepares Q/K at local head zero or a global three-head group boundary, reuses
the same prepared vector while that group continues, and releases it at either
the global group end or the final local head. Raw Q/K inputs are still consumed
on every head, so reader ring counts remain unchanged. The same input values,
normalization and per-head recurrence instructions are used; there is no
cross-core sharing or new synchronization.

The B32 tradeoff is explicit: maximum assigned heads falls 15 to 14, total Q/K
preparations rises 512 to 585, and maximum preparations on one core rises five
to six. The hypothesis is that removing one state recurrence from critical
cores outweighs the extra vector preparation. Timing can reject this hypothesis;
no speedup is claimed. B1 and disabled grouping retain their original partition.

Frozen files:

- `candidates/gdn_sfpu_partial_group_balance.py`
- `candidates/integrated_partial_group_balance134.py`
- `candidates/make_gdn_partial_group_balance_candidate.py`
- `candidates/check_gdn_partial_group_balance_cpu.py`
- `candidates/gdn_partial_group_balance_cpu.json`

CPU/source checks passed all 7,040 combinations of B1–32, requested core counts
1–110 and grouping enabled/disabled. Every head is covered once, prepared Q/K
always belongs to the current global group, all CB preparation/release counts
balance at core boundaries, and every raw input is consumed. Reader/writer and
the complete per-head state traversal are unchanged. Python compilation passed.

Parent validation uses the existing recurrence comparator against
`gdn_sfpu_packed_prepared_qk.py` with both persistent-state dtypes. Default B3
and B32 exercise partial groups; `--batches 3 --cores 5` additionally exercises
longer uneven local ranges. Require exact output/state bits before B32 short/512
and B1 short. Hardware correctness and latency remain unverified. No hardware
or main/shared source was touched.
