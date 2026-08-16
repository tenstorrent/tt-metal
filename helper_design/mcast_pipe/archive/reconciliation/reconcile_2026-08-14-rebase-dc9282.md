# Reconcile report — `mcast_pipe` after rebase — 2026-08-14

## Scope

- Branch: `sjovic/mcast-migration`.
- Pre-rebase HEAD: `40692d1e9ee4beb2a2db4196e1718407bf254bc4`.
- Old baseline: `4a1d6a97ca9bd4efabd0ad6115fcb30538851c90`.
- New fetched baseline: `dc9282be7d5e9d5a4b9137c1bf327de8d923e18e`.
- Rebased HEAD: `91bf395736241f7b62941c19383028e4f53da2ad`.
- Backup: `backup/mcast-migration-prerebase-20260814-40692d1`.
- API state: ledger v10, helper header v11; neither version changed here.

The user explicitly requested a fully automated rebase and autonomous decisions, which replaced the
skill's normal reconciliation mutation gate. The itemized rebase choices are archived in
`rebase_decisions_2026-08-14-dc9282.md`. This reconciliation itself was static: it changed rollout
paper state but did not migrate a kernel or advance a rollout status.

## Reconciliation buckets

| Bucket | Count | Result |
|---|---:|---|
| unchanged | 79 | Recorded paths exist and require no migrated-state action |
| added | 0 | No new intra-chip multicast-pipe candidate survived source inspection |
| removed | 0 | All 91 recorded kernel paths exist |
| renamed | 0 | No recorded path moved |
| clobbered | 0 | All 17 migrated kernels still reference `mcast_pipe` |
| rebase-touched | 12 | Helper use remains intact; `needs_recheck` added or retained |

Status counts remain 17 migrated, 3 pending, 71 deferred, and 0 quarantined. No entry lost rollout
history, no status changed, and no annotation file was needed.

## Rebase-touched migrated kernels

The baseline diff and recorded behavior-overlap resolutions identify these verify-only follow-ups:

- Sort: `coordinator_single_row_multi_core.cpp`, `reader_single_row_multi_core.cpp`.
- Matmul in1: `reader_bmm_tile_layout_in1_sender_writer_padding.cpp`,
  `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp`.
- GroupNorm v2: legacy sender/receiver and Welford sender/receiver.
- LayerNorm pre-allgather: sender and receiver.
- TopK: `reader_final_topk.cpp`, `writer_local_topk.cpp`.

The existing Matmul flags were retained and the other ten rows gained `needs_recheck`. Their status
remains `migrated` at ledger API v10.

## Recall sweep

The recognition family was copied from `design/primitive_contracts.md` and evaluated at the
pre-rebase and rebased heads. Raw-hit paths changed from 233 to 235: nine appeared and seven
disappeared. None of the disappearing paths was inventoried.

The nine appearing paths were read individually:

- `worker_sync_utils.hpp`: generic remote unicast semaphore increment;
- ring-attention halo reader/writer: fabric readiness and local flushes;
- strided-all-gather signal aggregator: unicast worker aggregation;
- high-bandwidth all-gather reader: generic dynamic semaphore wrapper;
- two fabric-bound minimal-matmul kernels: fabric peer handshakes;
- fabric-bound matmul common header: ordinary unicast-write flushes;
- generated `mcast_topology.py` example source: a final unicast-write barrier.

None contains an intra-chip multicast data/flag block in the helper's rollout boundary, so the
recalled candidate count is zero.

## Artifact changes

- `migration/ledger.json`: baseline updated, ten `needs_recheck` flags added, all 31 current migration
  commit references remapped through the range-diff, the current LayerNorm builder-prerequisite flag
  remapped, and reconciliation history appended. Historical `verified_at_commit` evidence remains at
  the tree where its tests actually ran.
- `migration/ledger.md`: mirror refreshed for the new head/baseline and 12 verify-only rows.
- `migration/test_map.json`: baseline and baseline note refreshed; inventory unchanged.
- Helper API and all kernel rollout statuses: unchanged.

## Existing post-rebase evidence

The enclosing rebase workflow, separate from this static reconciliation, passed:

- `./build_metal.sh`;
- exact and expanded Sort checks (six expanded cases);
- `test_mcast_pipe.py` plus source audit, 97/97;
- `McastHostFixture`, 32/32;
- focused Matmul, TopK, GroupNorm Welford, and LayerNorm pre-allgather operation probes, 4/4.
- complete operation coverage for all 12 rebase-touched kernels: 1,228 passed, 190 expected skips,
  90 expected xfails, and no failures or hangs.

These focused results establish that the conflict-composed tree builds and its highest-risk paths run.
They do not replace the complete mapped inventories owned by `apply-dm-helper`.

## Follow-up

Re-invoke `apply-dm-helper helper_design/mcast_pipe --mode=halt`. It should verify-only the 12
`needs_recheck` kernels using their complete mapped inventories and clear each flag when green. The
three pending and 71 deferred entries remain otherwise unchanged.

Before any publication, separately resolve the inherited `tt_ops_code_gen` gitlink at `4860704b`,
which is unavailable from its remote. Also track the pre-existing Matmul in0 block-sharded partial-grid
acknowledgement mismatch found during the independent consensus audit; it was not introduced by this
rebase and does not change the reconciliation buckets.
