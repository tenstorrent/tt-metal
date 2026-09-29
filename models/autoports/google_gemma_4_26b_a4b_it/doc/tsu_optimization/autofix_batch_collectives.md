# Shared-batch collective contract follow-up

The independent inspection in `autodebug_batch_collectives.md` found no ordering
race on the supported Linear TP4, single-command-queue, separate RS/AG path.
The misleading pool comment was corrected, and the new batching path explicitly
guards Linear topology. No extra synchronization or kernel change was required.

The follow-up guard/reuse CPU suite passes48 tests. Final B32 and repaired-control
default B8 Watcher/allocation contracts each pass six exact output/KV cases.
Opt-in and final-default serving each match104 benchmark completions exactly;
18 concurrent qualitative replies match the control. Default C32 TSU improves
5.70%, C8 improves2.86%, and C1 is unchanged. Independent bounded local stage
review is clean-pass. New exact-image/fullmatrix qualification remains pending.

No claim covers Ring, multiple queues, concurrent callers or fused collectives.
No hardware reset, foreign process termination or SWE evaluation change occurred.
