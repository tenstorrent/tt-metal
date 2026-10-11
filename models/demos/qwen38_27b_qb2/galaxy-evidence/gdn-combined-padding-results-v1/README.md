# Combined GDN model results and completed phase capture

Full-model comparison completed October 11, 03:12:38 UTC with clean teardown.
The resident-state/compact-gate/padding-skip candidate preserved precisions and
produced bit-identical output tokens against the matched compact controls.
Each workload used a warmup and three repetitions, B16 on one physical TP4.

| Context | Mean of control TSU | Candidate TSU | Step time | Gain |
|---|---:|---:|---:|---:|
| 32K | 19.9952 | 20.8846 | 47.8823 ms | 4.448% |
| 16K | 22.7214 | 23.8762 | 41.8826 ms | 5.083% |

This is measured decode performance; it is not a newly qualified GPQA result
or eight-replica saturation measurement. The 128-output-token comparison does
not replace model evaluation. Control drift was 0.00293% at 32K and 0.00172% at 16K.
The earlier 4096-update B16/B32 component correctness check also passed.

The original follower skipped expensive qualification because the primary gain
was 0.8894 TSU, below its 1 TSU batching gate. The user subsequently set the next
full-GPQA gate to measured 25 TSU at B16/32K/TP4. That policy is implemented in
run_compact_followup.py and does not alter these frozen experiment receipts.
The final decode target remains 30 TSU. No serving promotion was made.

The dependent current-kernel phase diagnostic completed at 03:13:55 UTC, with
all eight cases, 48 targeted kernel calls, four distinct physical ranks and
all 24 required phase labels. It passed independent dense recurrence checks,
exact plain/annotated and zero/skip comparisons, finite outputs, zero public
padding and clean teardown. Phase timing analysis and hardware-counter capture
remain separate work; no physical DRAM utilization claim follows from this pass.

Full raw device CSV (445,029,190 bytes) remains on the host; capture.json records
its path/hash. The much smaller ops CSV, phase receipt, final queues and full
before/candidate/after sweep receipts are retained here. XML/CSV are compressed
without altering their content. SAFE_PYTEST emitted an ops-CSV discovery warning
although native Tracy generated the retained CSV; the controller separately
verified raw device records and all phase labels. No reset recovery was needed.

Image/video implementation continues in separate feature worktrees. Prefix/SSD
restoration is still standalone and not enabled in serving; AgentX stays gated
on both features being integrated. See the experiments logbook for the policy
updates and launch evidence for immutable sources and exact reproduction args.
