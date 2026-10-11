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
padding and clean teardown. Phase timing analysis is retained in phase-analysis.json. Hardware counters
remain uncollected; no physical DRAM utilization claim follows from this pass.

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


## Current kernel phase attribution

The parser paired 3,460,608 selected raw events with exact device/call IDs and
signposts. All expected per-item/per-core phase counts match; all intervals
close. The retained analyzer reproduces the result from the full raw CSV and
ops CSV at their paths in the capture manifest.

B16 padding-skip eager kernel medians are 79.641 us recurrence and 31.536 us
epilogue. These are synthetic components with the native profiler active,
not full-model traced latency. The equivalent annotated medians are 79.579 us
and 32.053 us; this compares custom-zone overhead, not native profiler on/off.
Three calls per rank are insufficient to interpret the small negative
recurrence difference as an improvement.

| Phase | Processor | Mean accumulated us per active core |
|---|---|---:|
| Recurrence DRAM issue/wait | Reader | 18.98 |
| Recurrence L1 preparation | Reader | 32.98 |
| Recurrence input wait | Unpack | 12.21 |
| Recurrence delta | Math | 39.93 |
| Recurrence state update | Math | 13.38 |
| Recurrence output reduction | Math | 12.26 |
| Recurrence register wait/packing | Pack | 50.70 |
| Epilogue input DMA | Reader | 0.94 |
| Epilogue formatting | Reader | 4.09 |
| Epilogue mean square | Math | 7.25 |
| Epilogue weight/gate | Math | 9.93 |

Processor phases overlap. Pack's 50.70 us includes waiting for math to publish
DEST; it is not 50.70 us of pure packing. Likewise Math/Unpack zones contain
synchronization. These observations motivate counter-backed math/formatting
work and overlap, not a claim of a saturated or congested NoC. The reusable
reader's 33 us formatting is partly hidden and cannot simply be subtracted from
the kernel. Hardware activity counters and a matched streaming ceiling remain
needed. Output projection and attention remain larger whole-step targets;
GDN component improvements alone cannot be projected to the 30 TSU goal.
