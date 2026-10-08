## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml: the rules, the metric, allowed paths.
- $HISTORY (history.md): every round's table, leaderboard, and next-steps.
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp: the W_GAMMA streamed-chunk loop. `poll_stick` runs only
  between issues and chunks. `async_read_barrier<TXN_ID>` blocks and gates the push. This is the change site.
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp (lines ~370-410): compute packs and pushes the row-0 stat
  (stats_transposed_local_cb) before the x*gamma pre-pass waits on weight_cb. So the spin can't deadlock.
- tt_metal/hw/inc/api/dataflow/noc.h:640 `Noc::is_read_trid_flushed`, and dataflow_api.h
  `noc_async_read_barrier_with_trid`: the barrier is the same predicate in a loop, plus invalidate_l1_cache.

## Nodes consulted
- r05-b01-a02 (parent): the reflection's push-gating diagnosis and its #1 fix.
- r05-b03-a02: the same diagnosis on the other 4-wave node, with the `is_read_trid_flushed` variant.
- r05-b01-a01, r05-b03-a01: the 2-wave wins, their timelines, and the B-push gate.
- r04-b01-a01 / r03-b04-a03: where the streamed gamma chunks came from.
- r03-b01-a03, r04-b03-a01: earlier push-path changes.
- Classification sections of all 53 nodes, read to avoid repeats: HiFi2 is forbidden, dual-NoC drain is settled,
  PRE-stat reformulations are neutral, multicast release is neutral.

## Docs / external references
- none
