# Reconciliation report — helper-design document audit — 2026-08-23

Approval: the user approved the exact reconciliation and archival proposal
before mutation. This was a static audit: no production migration, build, or
device test was performed.

## Result

| Bucket | Count | Detail |
|---|---:|---|
| unchanged | 104 | every prior ledger path still exists |
| added | 4 | TT-Train Frobenius reader and K-split sender/two receivers |
| removed | 0 | no recorded call site disappeared |
| renamed | 0 | no path rename detected |
| clobbered | 0 | all 31 migrated kernels still use `mcast_pipe` |
| rebase-touched | 0 | this was not a baseline move |

The reconciled ledger contains 108 kernel rows: 31 migrated, 2 pending, and 75
deferred. Its 32 host-binding rows remain 27 migrated and 5 pending.

## Added call-site inventory

- `tt-train/sources/ttml/metal/ops/frobenius_normalize/device/kernels/dataflow/reader_frobenius_normalize.cpp`
- `tt-train/sources/ttml/metal/ops/k_split_gram_matmul/device/kernels/mcast_sender.cpp`
- `tt-train/sources/ttml/metal/ops/k_split_gram_matmul/device/kernels/mcast_receiver.cpp`
- `tt-train/sources/ttml/metal/ops/k_split_gram_matmul/device/kernels/mcast_receiver_writer.cpp`

`tt-train/sources/ttml/metal/common/dataflow_utils.hpp` defines a shared
loopback semaphore-broadcast utility. No call site for that function was found
in the current tree, so it is classified as support-only rather than a
standalone operation row.

The Frobenius and K-split program factories are mapped in `test_map.json` as
discovery-only deferred bindings. They are not added to ledger
`host_bindings`, whose established contract contains only migrated or
source-integrated pending bindings.

## Document reconciliation

The obsolete live rollout plan, aborted-rebase conflict record, original
intent, and superseded helper proposal moved into the archive. Live README,
ledger, test map, feedback tracker, design headers, audit summary, and dated
annotation/log status wording were updated. Historical measurements and prior
API outcomes were preserved rather than rewritten as current evidence.

## Static validation

- all ledger kernel and host-factory paths exist;
- all migrated kernels retain direct helper use;
- exact multicast-specific recall converges after the four additions, with
  substrate and support-only paths classified explicitly;
- JSON parses and inventory counts are internally consistent;
- all local Markdown links resolve.
