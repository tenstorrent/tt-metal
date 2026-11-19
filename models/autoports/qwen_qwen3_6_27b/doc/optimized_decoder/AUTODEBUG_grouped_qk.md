# Immutable candidate: reuse prepared Q/K for three value heads

Prototype: `candidates/gdn_sfpu_grouped_qk.py`, derived by snapshot from `gdn_sfpu_bf16_state.py`. No hardware execution, main decoder change, shared-kernel edit, or performance claim. Source SHA256 and CPU checks are in `grouped_qk_cpu_checks.json`.

The public function remains `recurrent_sfpu_compact(q,k,v,g,beta,state,...)`, with an optional `grouped_qk=True` switch. All original raw TILE, separate RM, packed RM, normalized TILE, FP32-state, and BF16-state interfaces are retained. `grouped_qk=False` selects the original partition and per-head preparation. Grouping is enabled only for B>=3; B1/B2 keep the original partition, preserving48 B1 cores rather than reducing to16.

For grouped execution, host work units are the16*B key-head groups. Each core receives a consecutive whole number of groups: global start and count are both divisible by3. Compute compile-time argument2 carries `GROUP_QK`; reader/writer arguments and source are unchanged. The compute local head0 corresponds to a group boundary. Each group's first head executes the original `prepare_vector` twice, producing CB13/14. The following two heads wait/pop their duplicate raw Q/K from CB0/1 without recomputing normalization/transposition. CB13/14 stay resident until the third head completes. The subsequent group can then reserve the same four slots.

Safety/source invariants:

- Three consecutive global heads `3*k,3*k+1,3*k+2` have the same user and key head because48 is divisible by3; reader mapping is `user=h/48`, `hk=(h%48)/3`.
- Prepared Q/K are read-only during the recurrence. Holding their CB pages changes neither arithmetic nor per-head normalization values.
- Every raw Q/K push is still matched by a pop; the reader's one-head rings can progress. Prepared Q/K have one push/pop pair per group and maximum occupancy4.
- All state/g/beta/value waits and pops, FP32 recurrence operation order, output packing, BF16 state rounding after current readout, and writer barriers remain exactly the original source.
- Every state head and output row has exactly one owner. Changed core assignment does not change the addresses;64-byte output row writes remain disjoint. The final global head still owns output padding.
- Reader traffic is deliberately unchanged. This first prototype removes compute work only; skipping redundant Q/K DRAM fetch/widening is a separate future change.

B32 on110 cores has512 groups:72 cores process5 groups/15 heads and38 process4 groups/12 heads. The old partition's maximum was14 heads. The new maximum state/readout work rises7.14%, while Q/K preparation calls fall from3072 to1024 across the chip (two-thirds removed). Whether the tradeoff wins must be measured. B3 uses48 cores instead of110 and may lose to reduced parallelism; use the explicit switch for a legal same-source control and choose the measured batch crossover afterward.

Validation completed without importing TTNN: Python syntax compilation; AST-isolated host partition and prepared-CB occupancy checks for1536 combinations covering B1..32, available-core limits, requested-core limits, and grouped on/off; exact string comparison of reader, writer, and retained-DST per-head update/readout code. These do not replace kernel compile, device correctness, watcher, trace, or latency validation.

Parent device validation should compare grouped on/off with identical inputs and restored state, first short outputs/state (expect bit-identical), then existing512-step fixture and traced warmed latency. Both controls must preserve the same acceptance results; this optimization does not repair the remaining `.992493` precision failure. Keep only if final correct-path latency improves.
