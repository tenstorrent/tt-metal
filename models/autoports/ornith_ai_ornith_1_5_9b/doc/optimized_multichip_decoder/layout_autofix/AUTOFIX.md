# AutoFix: producer and consumer shard agreement

Failing performance contract: matmul output placement and following input plans
must agree where possible; row projection must not reshard only to satisfy an
unnecessary all-reduce caller assumption. Accuracy remains at least0.995.

AutoDebug confirmed that the op computes actual output cores from N/per_core_N
and packs those over11 columns. It does not honor a requested rectangle unless
an optional output tensor is supplied. The CCL already supports different input
and output shard grids. The warning itself does not move data.

Prioritized repairs:
1. Consume native norm shards for K4096 projections, actual gate/up output shards
   for down, and let direct AR read row-projection shards without a separate
   reshard. Keep the32-core residual grid, workspace and semaphore ownership.
2. If needed, preallocate optional matmul outputs with explicit lifetime ownership.
3. Broader64-core residual contract, including norm/program/workspace updates.

Implemented minimal option1 behind construction-time movement_optimized (default
True); False retains the same integrated grouped code with the old movement plan
for an A/B control. Output metadata now describes computed storage. No shared
kernel or numerical policy change. Evidence pending: same-harness both kinds and
batch1/32 A/B, actual layouts, profiler movement counts, tracked replay and watcher.
Not yet claimed verified.

## Focused device validation

Same-harness movement_control_stack_{1,32}.json versus movement_stack_{1,32}.json
both pass all prefill/continuation/decode comparisons. B1 improves0.56903 to
0.55806ms; B32 changes9.06102 to9.05500ms (not a material speed claim for B32).
Minimum PCC is unchanged at0.9999326 for B1 and0.9997884 for B32.
Exact commands and complete source snapshots are recorded by run_stage5_jobs.
Full final default trace-tracking, stress, context and watcher gates remain pending.
