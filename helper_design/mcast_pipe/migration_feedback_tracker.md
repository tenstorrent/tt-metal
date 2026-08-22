# `mcast_pipe` migration feedback resolution tracker

Started: 2026-08-22

This tracker is the execution record for
[`migration_feedback.md`](migration_feedback.md). Items advance one at a time.
Each item is complete only after its source changes, cross-operation audit where
applicable, focused validation, and the migration guardrail checks pass.

## Guardrail checklist used at every gate

- Preserve multicast behavior, receiver geometry, synchronization policy, and
  semaphore ownership.
- Keep helper compile-time and runtime blocks opaque and contiguous.
- Derive every post-helper offset; do not encode helper block widths.
- Audit TensorAccessor bases, optional tails, fused-operation fields, descriptor
  paths, legacy paths, and cache overrides after ABI changes.
- Add no migration-only preprocessor branches and keep diffs focused.
- Run device tests sequentially through `scripts/run_safe_pytest.sh` after first
  proving one focused parametrization compiles and passes.

## Progress

| Order | Feedback | State | Cross-operation scope | Validation |
|---:|---|---|---|---|
| 1 | MATMUL-001 | Complete | 1D and 2D, descriptor and legacy producers | Build; focused 1D `--dev`; focused 2D; 22 source audits passed |
| 2 | MATMUL-002 | Complete | Symmetric inactive operands in both 1D variants; shared optional decoder | Build; focused `MCAST_IN0` `--dev`; focused `MCAST_IN1`; 22 source audits passed |
| 3 | MATMUL-003 | Complete (premise corrected) | 1D descriptor/legacy `MCAST_IN0` and `MCAST_IN1` divergent partial rectangles | Exact uneven-width hang reproduced; override restored; build and exact `--dev` rerun passed |
| 4 | MCAST-001 | Complete | All ledger-migrated/source-integrated kernels and host bindings | Builds; focused family device gates; 22 source audits passed |
| 5 | MCAST-003 | Complete | Host helper API and every migrated producer | Build; append API gtest; 23 source audits; focused family device gates |
| 6 | MCAST-002 | Complete | Matmul, Conv2D, Argmax, and cross-operation Group Attention mixed roles | 24 source audits; helper 80/80 `--dev`; four focused operation gates |
| 7 | MCAST-004 | Complete | Tagged helper ABI, all migrated host/kernel consumers, inactive Matmul operands | Build; 25 source audits; helper 80/80 `--dev`; host gtest 36/36; 1D in0/in1, Sparse, and chained 2D gates |
| 8 | MATMUL-004 | Complete | 1D and matching 2D legacy/descriptor common in0 tails | Build; 26 source audits; focused 1D and 2D `--dev` gates |

## Evidence log

### MCAST-004 (complete)

- Confirmed this is a caller-facing, fleet-wide ABI change: every present
  `Mcast1D`/`Mcast2D` compile-time block gains a leading true tag, while the
  inactive shared-Matmul blocks emit only a false tag and no runtime payload.
- Added the tag to both host serializers and specialized kernel decoding so an
  absent block consumes one compile-time word, no runtime words, and cannot
  produce either pipe face. Removed `OptionalMcastArgs` and all Matmul-owned
  presence flags; five inactive Matmul/Sparse bindings now append the opaque
  absent block.
- `./build_metal.sh`, 25 source audits, all 80 helper device tests under
  `--dev`, and all 36 `McastHostFixture` gtests passed. Sequential device gates
  passed for both asymmetric 1D Matmul directions, Sparse Matmul, and chained
  2D Matmul, covering absent and present blocks plus derived following offsets.

### MATMUL-004 (complete)

- Moved the common in0 helper append after the sharded/interleaved conditional
  in both 1D builders. The cross-operation audit found and fixed the same
  genuinely identical tail in both 2D builders; no other migrated producer had
  this pattern.
- Added a source regression test that requires exactly one append after each of
  the four conditionals. `./build_metal.sh`, all 26 source audits, and focused
  1D and 2D Matmul gates under `--dev` passed.

### MCAST-003 (complete)

- Added append-style compile-time and per-core runtime APIs to `Mcast1D` and
  `Mcast2D`. The common implementation preserves an opaque helper wire for both
  `std::vector<uint32_t>` and descriptor runtime vectors with buffer variants;
  a focused host gtest verifies the appended block exactly matches the getter
  form.
- Converted every migrated host producer to complete operation-owned branch
  prefixes before a single helper-tail append. This covers all Matmul legacy and
  descriptor paths, Sparse Matmul, Group Attention Matmul, Conv2D and Conv3D,
  GroupNorm and LayerNorm, Argmax, TopK, Sort, Move overlap, and SDPA decode.
- Added a ledger-wide audit that permits only three deliberate getter-query
  sites and rejects production bindings that do not emit both helper wires via
  append APIs. The full audit passed 23/23.
- `./build_metal.sh` passed after the conversion. The focused append gtest and
  sequential device gates passed for TopK, Argmax, Sort, both Move layouts,
  Matmul 1D/2D/Sparse, GroupNorm, LayerNorm, SDPA, Conv2D width/block sharding,
  Conv3D, and Group Attention Matmul. LayerNorm, Conv2D width, and Conv3D each
  encountered the known unrelated Watcher C++17 `ASSERT` compile incompatibility
  in another kernel and passed the identical safe-wrapper parameter without
  `--dev`.
- The Group Attention gate exposed a pre-existing mixed-role construction bug:
  sender-only cores eagerly constructed a receiver and tripped the helper's
  `can_receive()` assertion. The MCAST-002 alias API made role-conditional
  optional storage possible; after applying it, the same exact `--dev` node
  passed. Role assertions were retained.

### MCAST-002 (complete)

- Exposed `McastArgs::SenderPipe` and `McastArgs::ReceiverPipe` as the ordinary
  concrete face types, with `SenderPipeFor<NOC_ID>` retaining explicit-NoC
  specialization. `sender(noc)` and `receiver(noc)` return those aliases.
- Replaced expression-based `decltype(...sender/receiver...)` storage in
  block-sharded Matmul, width-sharded Conv2D, and Argmax. The ledger-wide audit
  found no remaining migrated occurrence and now prevents regressions.
- Applied the same API to Group Attention Matmul. Triage of the pre-fix exact
  node showed sender-only cores stopped at `ASSERT(can_receive())`; both pipe
  faces now live in optional storage and are constructed only for permitted
  roles. The same exact node passed under `--dev` after the fix.
- Focused block-sharded Matmul and two-rectangle Argmax nodes passed under
  `--dev`. Width-sharded Conv2D hit the known unrelated Watcher C++17 `ASSERT`
  compile incompatibility in `eltwise_typecast`, then passed the identical safe
  parameter without `--dev` at PCC 0.99823. The complete helper device suite
  passed 80/80 under `--dev`, and the expanded source audit passed 24/24.

### MCAST-001 (complete)

- Generated the scope from the migration ledger and paired migrated kernels
  with their host producers; families already conforming are retained without
  churn.
- Reordered the interleaved and sharded GroupNorm positional CT/RT prefixes
  ahead of their stable three-block multicast tails. Host build passed; exact
  sharded legacy 8x4 device node passed under `--dev` after its compile gate.
- Reordered distributed LayerNorm pre/post-allgather CT/RT builders and added
  named per-role runtime boundaries for variable coordinate tails. Host build
  and an exact post-allgather sender/receiver node under `--dev` passed.
- Reordered TopK final-reader and local-writer operation CT/RT prefixes ahead
  of their multicast tails. The all-operation audit then found Argmax's two
  helper blocks preceding its input/output TensorAccessor descriptors; those
  descriptors now complete the operation CT prefix before both helper blocks.
  Host build and exact multicore TopK and two-rectangle Argmax nodes under
  `--dev` passed.
- Reordered SDPA decode operation fields, output-core coordinate arrays, and
  TensorAccessor blocks ahead of the K multicast tails. The idle-core runtime
  allocation now derives the helper width from the helper object rather than
  spelling it. Host build and an exact replicated-Q MLA decode route under
  `--dev` passed, including three iterations and cache reuse.
- Reordered Conv2D width-sharded operation CT fields, its optional config
  TensorAccessor descriptor, and operation RT fields ahead of the activation
  multicast tails. Both optional branches now have one stable ABI boundary by
  emitting a null accessor descriptor when config tensors stay in L1. Host
  build, the exact non-DRAM width-sharded node under `--dev`, and the matching
  `config_tensors_in_dram=True` node passed.
- Reordered Conv2D height- and block-sharded weight-sharing RT layouts so
  `remaining_tiles_to_push`, `is_sender_core`, and `skip_work` finish stable
  operation prefixes before the helper tails. Host build and exact height- and
  block-sharded nodes passed under `--dev`.
- Reordered all Matmul block-sharded, interleaved in0/in1 sender, receiver,
  sparse, descriptor, and legacy compile/runtime producers to operation-first,
  helper-tail layouts. Variable fused-operation runtime prefixes use an
  explicit runtime boundary when constructing `McastArgs`; cache rebinding
  indices were updated to operation-owned positions.
- The exact uneven-width 1D MCAST_IN0 gate initially hung after the earlier
  MATMUL-003 edit. Triage proved its bounding box contains inactive landing
  cores, so `num_cores - 1` is a required divergent ACK count. Restoring it in
  legacy/descriptor MCAST_IN0 and MCAST_IN1 builders made the same exact
  `--dev` node pass.
- Sparse Matmul exposed and then validated three additional cross-path fixes:
  named optional-helper bindings, receiver batch CT fields before its helper,
  and removal of the inactive synthetic in1 helper family. After rebuild, the
  exact `test_sparse_matmul_without_nnz` compile/correctness node passed under
  `--dev`.
- Block-sharded and ordinary 2D in1 exact nodes also passed under `--dev`.
- Conv3D's variable worker-coordinate RT prefix now completes before its helper
  block and uses the helper's explicit runtime-base constructor. The host build
  passed; the exact compile-focused test was expectedly skipped under Watcher
  (issue #37184), then passed through the safe wrapper without `--dev`.
- The final ledger-wide audit now checks natural source order for positional
  operation arguments and passed 22/22. Exact GroupNorm legacy 8x4 and
  distributed post-allgather LayerNorm nodes were rerun under `--dev` after the
  source-order cleanup and passed.

### MATMUL-001

- Reordered all six operation compile-time arguments ahead of the opaque in0
  multicast block in the receiver kernel and all four 1D/2D descriptor/legacy
  host producers. The runtime prefix is empty, so its helper-tail ordering was
  already correct.
- `./build_metal.sh`: passed.
- Focused 1D interleaved receiver/cache-reuse node under `--dev`: 1 passed.
- Focused 2D interleaved receiver/cache-reuse node: 1 passed.
- `test_mcast_pipe_source_audit.py`: 22 passed.

### MATMUL-002

- Removed both synthetic one-core `Mcast2D` objects and their compile-time and
  runtime blocks from the 1D legacy and descriptor builders.
- Added `OptionalMcastArgs`: active variants retain the ordinary opaque
  `McastArgs` boundary, while inactive variants consume no helper words and do
  not instantiate a multicast decoder. The selection is a named compile-time
  binding, not a new preprocessor branch.
- Audited both runtime rebinding mechanisms after the ABI contraction. The
  legacy cache override and descriptor tensor bindings now share named
  `McastIn0RuntimeArgIndices` instead of the stale `[7]`/`[18]` positions.
- `./build_metal.sh`: passed.
- Focused 1D `MCAST_IN0` interleaved/cache-reuse node under `--dev`: 1 passed.
- Focused 1D `MCAST_IN1` interleaved/cache-reuse node: 1 passed.
- `test_mcast_pipe_source_audit.py`: 22 passed.

### MATMUL-003

- Initially removed the explicit count per the feedback, then exercised the
  required uneven-width route during MCAST-001. That exact route hung with the
  sender waiting in `SenderPipe::send()` while receivers waited for the next
  signal.
- Triage and factory geometry confirmed the operation supplies a bounding
  rectangle containing inactive landing cores, but only `num_cores - 1`
  receiver kernels acknowledge it. This is the helper's documented divergent
  count case, so the explicit override is required rather than redundant.
- Restored the override in both legacy and descriptor MCAST_IN0 builders and
  retained it in symmetric MCAST_IN1 builders, whose bounding boxes can also be
  partial. Sort and Conv2D retain their independently confirmed divergent
  overrides.
- `./build_metal.sh`: passed.
- Exact uneven-width 1D MCAST_IN0/cache-reuse node under `--dev`: 1 passed
  after restoration (and deterministically hung without it).
