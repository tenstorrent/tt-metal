# Initial independent stage review

Verdict: `more-work-needed`.

The fresh reviewer accepted the measured TP=4 path, PCC, persistent CCL,
non-aligned coverage, context execution, lower-movement AutoFix closure, and
watcher result, but found three evidence/documentation gaps:

1. The retained shard-advisor capture predated dedicated RoPE, direct output,
   and the stack-native residual. It requested a final-graph capture, a coherent
   application of all feasible choices, explicit blockers for unfixable ops,
   and a current-graph packed-projection rerun.
2. The profiler section mixed paths and did not reconcile wall time, device
   time, gaps, the 90 ms cross-iteration gap, or raw versus derived core-count
   semantics. It also requested an explicit byte/bandwidth roofline.
3. `doc/context_contract.json` did not describe the exact persistent-buffer and
   semaphore footprint, manager/owner lifetime, cleanup order, or inter-layer
   residual contract.

The reviewer also asked that the two-layer evidence use two decoder instances
sharing a pool instead of one decoder object reused twice. All items are
remediated in the post-review hardware artifacts, updated context contract,
`results/roofline_accounting.json`, and `shard_advise/final_graph/`.
