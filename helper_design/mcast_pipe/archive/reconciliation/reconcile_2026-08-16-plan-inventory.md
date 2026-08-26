# Reconcile report — approved plan inventory — 2026-08-16

## Scope and approval

- Branch: `sjovic/mcast-migration`.
- Plan-head source state: `db246d49b89978d436d944d57f0ba326ef698416`.
- Baseline: `origin/llk_helper_library` at `dc9282be7d5e9d5a4b9137c1bf327de8d923e18e`.
- API state: ledger v10, helper v11; neither changed during this static reconciliation.
- Approval: the user approved `helper_design/mcast_pipe/plan.md`, explicitly requested Tier 0, Tier 1,
  and Tier 2 units 7-9, and requested unattended execution.

The audit result exactly matched plan Appendix B. No unapproved record-losing or retagging delta was
applied. Claude reviewed the proposed schema; its corrections to family names, tags, annotation reuse,
and deferred host-binding treatment were incorporated before mutation.

## Reconciliation buckets

| Bucket | Count | Result |
|---|---:|---|
| unchanged | 91 | Every prior path exists; existing statuses preserved |
| added | 13 | Approved Appendix-B call-site/receiver companions |
| removed | 0 | No record lost |
| renamed | 0 | No path moved |
| clobbered | 0 | All 17 migrated kernels retain helper use |
| rebase-touched | 0 | This was inventory completion, not a baseline move |

Final paper state: 104 kernel entries — 17 migrated, 3 pending, 84 deferred — and 23 host bindings.

## Added inventory

### Matmul Decode — Tier 3 unit 18 prototype

- `reader_full_width_sharded.cpp`
- `reader_partial_width_sharded.cpp`

Both byte-identical readers are `matmul / hybrid / refactor-high`. Two hubs multicast disjoint regions
into the same destination CB and independently increment a shared count-of-two completion semaphore.
The entries remain deferred pending the plan's prototype-first composition gate; `refactor-high` avoids
prejudging that prototype as impossible.

### Programming and lab examples — D5

- Matmul `reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp`: `matmul / receiver / clean`.
- Contributed multicast `inbound_kernel.cpp`: `ccl / deepseek / examples / receiver / ref`.
- Lab multicast `mcast_receiver.cpp`: `ccl / deepseek / examples / receiver / ref`.

These are receiver companions of already-inventoried sender/mixed example kernels. D5 remains a scope
deferral, not a helper capability verdict.

### Quasar Matmul — D4

- In0 receiver original and `_metal2`: `receiver / refactor-high`, matching the current production
  receiver truth including its batch-valid `wait_min`/mailbox branch.
- In1 receiver/writer original and `_metal2`: `receiver / clean`.

All use family `quasar (experimental metal 2.0 port)`, retain `quasar-metal2-port`, and remain deferred.

### Quasar Conv2D — D4

- 1D weights receiver original and `_metal2`.
- 2D weights receiver original and `_metal2`.

All four are canonical clean receivers in the Quasar family and remain scope-deferred.

## Support-only atomic dependencies

These existing headers contain no multicast block or semaphore primitive and are intentionally not ledger rows:

- Production Conv `device/kernels/conv_reader_common.hpp` — D1 support.
- Quasar Conv `device/kernels/conv_reader_common.hpp` — D4 support.

They must still be audited with any future migration of their including kernels.

## Regenerated factory/host companion map

Deferred factories are documented here rather than appended to `ledger.host_bindings`, whose current
convention contains only migrated or source-integrated pending bindings.

| Added kernel family | Factory/host companion |
|---|---|
| Full-width Matmul Decode | `experimental/matmul_decode/device/full_width_sharded_program_factory.cpp` |
| Partial-width Matmul Decode | `experimental/matmul_decode/device/partial_width_sharded_program_factory.cpp` |
| Programming-example dual receiver | `tt_metal/programming_examples/matmul/matmul_multicore_reuse_mcast/matmul_multicore_reuse_mcast.cpp` |
| Contributed multicast inbound | `tt_metal/programming_examples/contributed/multicast/multicast.cpp` |
| Lab multicast receiver | `ttnn/examples/lab_multicast/lab_multicast.cpp` |
| Quasar Matmul in0 receivers | Quasar 1D/2D reuse-mcast factories; sparse factory for the original path; Metal2 create paths select `_metal2` |
| Quasar Matmul in1 receivers | Quasar 1D/2D reuse-mcast factories; Metal2 create paths select `_metal2` |
| Quasar Conv weights receivers | `experimental/quasar/conv2d/device/conv2d_op_sharded_program_factory.cpp` |

Basename ambiguity was resolved by reading each factory's kernel base/path selection; the Quasar unqualified
names bind under the Quasar kernel base, not the production Conv/Matmul directories.

## Annotation and validation

- New protocol annotation: `kernel_annotations/matmul_decode_two_hub_readers.md`.
- Existing family annotations extended for programming examples, lab multicast, Quasar Matmul, and Quasar Conv.
- JSON validation: 104 unique paths, no duplicate kernel key, 13 additions dated 2026-08-16.
- Static only: no device test, build, production source edit, or helper-version change occurred here.

## Apply hand-off

Proceed with the approved rollout order:

1. Verify/stamp the migrated v10 fleet at helper API v11 and clear 12 `needs_recheck` flags.
2. Verify Tier 0 units 1-2; unit 1 write-back remains blocked on separately authorized historical matched
   performance acquisition.
3. Apply Tier 1 unit 6.
4. Apply Tier 2 units 7-9 only.
