# Reconcile report — `mcast_pipe` — 2026-08-14

## Scope and approval

- Branch: `sjovic/mcast-migration` at `9686814ea22`.
- Rollout baseline: `origin/llk_helper_library` at `4a1d6a97ca9` (confirmed ancestor).
- Paper-state comparison point: `fae7eb9ed6f`, the last commit that updated both the former text inventory and ledger.
- User approved the itemized reconciliation diff before artifact mutation.
- This pass was static: no source migration, build, device test, or helper API-version change was performed.

Post-reconciliation documentation cleanup completed the binding inventory that
this report handed off: four block-sharded Matmul legacy/descriptor bindings and
one block-sharded Conv activation binding were added as `pending` to both
`ledger.json` and `test_map.json`. Current host-binding counts are therefore 14
migrated-at-v10 and 10 pending. No migration status was advanced.

## Reconciliation buckets

| Bucket | Count | Result |
|---|---:|---|
| unchanged | 88 | Paths exist; migrated rows still use `mcast_pipe`; no post-snapshot source edit |
| added | 0 | No recognition-family path was added after the paper-state snapshot |
| removed | 0 | All 91 recorded call-site paths exist |
| renamed | 0 | No source path was added or renamed |
| clobbered | 0 | Every migrated kernel still references the helper |
| rebase-touched | 3 | Helper use remains intact; verify-only evidence is required |

At reconciliation time, the former text inventory contained 91 unique paths and
exactly matched the 91 unique kernel paths in `migration/ledger.json`. The ledger
is now the sole inventory.

## `needs_recheck`

These migrated API-v10 kernels changed after `fae7eb9ed6f` while retaining helper use:

- `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` — changed by Matmul remediation and naming cleanup;
- `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` — changed by Matmul naming cleanup;
- `activation_reader_width_sharded.cpp` — changed by the streaming-overlap optimization and its measured-win gate.

Their ledger status remains `migrated`; only the advisory `needs_recheck` flag was added.

## Resolved deferrals retagged to pending

Two existing inventory entries had stale design-gap classifications even though their source now uses the helper:

- `reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` — the API-v11 host model now
  represents the ordered rotating sender set independently of receiver geometry;
- `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp` — `SenderPipe::send_from_cb`
  now represents the producer-overlapped streaming multicast behavior.

Both moved from `deferred` to `pending`. Reconcile does not claim that source presence is validation;
`apply-dm-helper` must verify them and write the final migration evidence.

The two existing Matmul in0 pending kernels and five pending Matmul host bindings already use API-v11
typed Flag control values. Their obsolete API-007 prerequisite flags were replaced with
`source-integrated` / `apply-verification-pending` while their status remained `pending`.

## Recall sweep

The recognition family from `design/primitive_contracts.md` was compared at `fae7eb9ed6f` and current HEAD.
It gained zero product-source paths and lost one raw-pattern hit: the block-sharded Matmul kernel stopped
using the open-coded primitive after adopting `mcast_pipe`. No code file was added or renamed in the
comparison range. The four source-integrated pending kernels were already in the inventory, so no new
annotation file was required.

## Paper state at the reconciliation checkpoint

- Kernels: 17 migrated, 4 pending, 70 deferred, 0 quarantined.
- Host bindings at this checkpoint: 14 migrated, 5 pending. The subsequent
  documentation cleanup mapped five additional required pending bindings as
  described above.
- Three migrated kernels carry `needs_recheck`.
- Ledger API remains 10, while the materialized helper header is API 11.

The version mismatch is intentionally not repaired here: helper-version staleness and device validation
belong to `apply-dm-helper`, not reconciliation.

## `apply-dm-helper` handoff

1. Re-enter the 17 API-v10 migrated kernels and 14 migrated host bindings at API v11.
2. Verify-only the three `needs_recheck` kernels, then clear their flags when green.
3. Verify and write back the two interleaved Matmul kernels and five bindings.
4. Verify and write back the block-sharded Matmul kernel and its four mapped bindings.
5. Verify and write back the block-sharded Conv activation kernel and its mapped binding.
6. Record exact fresh-JIT, complete-inventory, host/helper, and performance evidence in the unit logs,
   ledger, report, changelog, and README.
