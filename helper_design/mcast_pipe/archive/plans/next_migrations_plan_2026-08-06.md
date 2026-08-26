# Archived: Next Mcast Host and Kernel Helper Migrations

Date: 2026-08-06 (revised after an `apply-dm-helper` conformance review)

Intended baseline: `llk_helper_library` at `4a1d6a97ca9`. Confirmed an ancestor of
the current branch head, so `census.txt` and `migration/ledger.json` are aligned
with the tree and no `reconcile-dm-helper` pass is required.

Original plan baseline: `MCAST_PIPE_API_VERSION = 10`. The execution checkpoint
below records the subsequent v11 bump.

## Execution checkpoint (stopped by request on 2026-08-06)

The plan has been executed through the API-007 helper change and is paused partway
through the required API v11 Tier 0 fleet revalidation. The repository is left at
`25263ce0bde` (`Add typed Flag signals to mcast_pipe v11`); no device test process
is running.

Completed work:

- Prerequisites and sparse coverage map: `40aa811f877`. The exact sparse Matmul
  without-`nnz` case was verified from a fresh JIT cache.
- Multicore TopK migration: `b5c99d43fd5`, with metadata correction
  `6277d3531e3`. The fresh-JIT case, complete mapped inventory (14 passed and 12
  expected xfails), host tests (25 at the time), and helper tests (77 at the
  time) passed.
- Sharded LayerNorm prerequisite split: `4ef7e9a57a6`.
- Sharded LayerNorm pre-allgather migration: `4acd98259b6`, with metadata
  correction `fae7eb9ed6f`. The exact fresh-JIT case and the complete
  `LN-PRE-ALLGATHER` (126), `LN-POST-ALLGATHER` (136), and `LN-SHARDED` (208)
  inventories passed, as did the host (28) and helper (77) suites.
- API-007 was implemented by bumping `mcast_pipe` to v11 and adding typed Flag
  signal values in `25263ce0bde`. Focused `VALID` and `IGNORE_BATCH` fresh-JIT
  tests passed. The complete helper suite passed 79/79 under `--dev`, and the
  host suite passed 28/28.

Tier 0 v11 revalidation status:

- Static caller audit found all 17 migrated kernels source-compatible with v11;
  no production-kernel rewrite is required. Four entries still need their
  `migration_unit` metadata corrected during write-back (two LayerNorm and two
  Sort entries).
- Gate 0 host (28/28) and helper (79/79) suites passed, and every selected exact
  fresh-cache JIT probe passed.
- Complete mapped inventories passed for Conv height/block/width (48 passed and
  16 expected skips each), Conv DRAM routes (3 configuration routes and 17 broad
  cases), TopK (14 passed and 12 expected xfails), Sort, all GroupNorm routes,
  LayerNorm pre-allgather, and LayerNorm post-allgather.
- Execution was stopped as the plain sharded LayerNorm inventory began. That
  inventory and the migrated Matmul-in1 inventories remain to run. Tier 0 ledger,
  report, test-map, dashboard, and API-version write-back has not started and
  must be performed only after those inventories pass.

Resume order:

1. Finish plain sharded LayerNorm and Matmul-in1 Tier 0 inventories sequentially.
2. Stamp the 17 kernels and 14 host bindings at API v11, correct the four
   `migration_unit` fields, update all paired rollout artifacts, audit them, and
   commit the Tier 0 unit.
3. Execute Matmul in0 multicast (item 3) against the v11 typed Flag API, including
   the host rebuild, exact 1D and sparse probes, full mapped inventories,
   block-sharded regression guard, helper/host suites, perf record, paired
   artifact write-back, and atomic commit.
4. Run the final source/artifact audits and record the completed rollout status.

## Recommended order

0. Prerequisites (below) — un-defer the scope, then per-unit prerequisites
1. Multicore TopK
2. Sharded LayerNorm pre-allgather
3. Matmul in0 multicast

TopK is the only unit that is ready to migrate as-is. LayerNorm pre-allgather and
Matmul in0 each carry a named prerequisite that must land and be validated before
their migration starts; those prerequisites are the real reason they rank second
and third, not helper-API distance.

Run mode for the whole sequence: **`--mode=halt`**. Two of the three units have
open structural questions, so a failure must leave its diff in the tree for
triage rather than being reverted into a summary line.

---

## 0. Prerequisites for the whole sequence

### 0.1 Un-defer the scope (blocking — `apply-dm-helper` stops without it)

The ledger at v10 holds 13 migrated kernels, 12 migrated host bindings, 78
deferred entries, and **zero `pending` and zero stale** entries. All six kernels
targeted by this plan are `status: deferred`. Gate 0's re-entry rule therefore
reports "rollout already current at v10" and stops, and migrating a `deferred`
entry is an explicit anti-pattern.

Before invoking the rollout skill, re-tag upstream:

- Record a resolution for each deferral reason. Only LayerNorm's is covered
  today (API-003, Implemented). There is **no API item** covering TopK's
  `race-free-no-handshake-init` flag or Matmul in0's `typed-control-values`
  flag; both decisions currently exist only in this document. Write them into
  `api_feedback.md` so the census re-tag has a citable resolution.
- Move the six kernels from `deferred` to `pending` in `census.txt`, and clear
  the `v9-port-blocked` / `design-gap` / `exact-baseline-restored` flags that no
  longer apply.
- Re-tag `reader_bmm_tile_layout_in0_sender_padding.cpp` and
  `reader_bmm_tile_layout_in0_receiver.cpp` from `clean` to **refactor-high**.
  Their current `clean` tag would sort Matmul in0 into the clean spine ahead of
  LayerNorm's `refactor` entries, inverting the intended order.

### 0.2 Gate 0 pre-flight (before touching any production op)

Run and confirm green *first*, not as post-migration validation:

- the `mcast_pipe` kernel-helper unit test;
- the host-only `mcast_host` tests (`tests/ttnn/unit_tests/gtests/test_mcast_host.cpp`);
- the end-to-end wire tests.

A red test here means the helper is not ready and the rollout does not start.

### 0.3 Close the sparse-matmul test-map gap (blocks item 3 only)

`migration/test_map.json` defines 13 inventories and **none covers sparse
matmul**; `host_bindings` likewise has no entry for
`sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp` (only the 1D/2D
legacy+descriptor in1 bindings are mapped). Item 3's riskiest change lands on
exactly that unmapped path.

Add a device-verified `MM-SPARSE-IN0` inventory and the sparse host binding to
`test_map.json` during Phase 1, before Gate A. Verification method is the usual
one: run the candidate parametrization and confirm the kernel appears in the JIT
build cache.

### 0.4 Execution conventions

- One atomic commit per migration unit, each **paired with its ledger
  write-back** at v10. Do not batch the ledger update to the end of the
  sequence; a `halt` stop must leave the ledger consistent with the tree.
- Rebuild with `./build_metal.sh` before device validation whenever host code
  changed.
- Activate `/localdev/sjovic/tt-metal/python_env/bin/activate` and run device
  tests through `scripts/run_safe_pytest.sh`, sequentially. Prove one exact
  parametrization under `--dev` from a fresh JIT cache first, then run the
  mapped inventory.
- Record a per-kernel perf delta against the `bakeoff_*` baseline in
  `migration/report.md`. This matters most for Matmul in0, which sits on the
  hottest multicast path in the tree.

---

## 1. Multicore TopK

Ready to migrate once 0.1 and 0.2 are done. No prerequisites of its own.

### Scope

- `ttnn/cpp/ttnn/operations/reduction/topk/device/topk_multi_core_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_final_topk.cpp`
- `ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_local_topk.cpp`

This is the complete atomic unit. The factory is the only binding site for both
kernels, and each kernel gets its own compile-time argument vector
(`reader_final_compile_time_args`, `writer_local_compile_time_args`), so no other
kernel's wire moves.

### Migration plan

- Build one host `Mcast2D` from the final coordinator to the local-worker
  rectangle. The coordinator sits **outside** that rectangle, so the object runs
  in sender-separate mode: fan-out equals the rectangle area, and the
  participating set is `rect ∪ {sender}` — which matches the factory's existing
  `all_cores_range_set`.
- Configure it with `handshake=false` and `DataReadyMode::Counter`.
- **Semaphore ownership:** the helper adopts the existing readiness semaphore via
  `cfg.sem_ids = {receiver_semaphore_id}` (id 1). `sender_semaphore_id` (id 0)
  stays operation-owned — it is the arrival counter, not a readiness signal. With
  adoption the helper creates nothing and the factory keeps both
  `SemaphoreDescriptor`s; confirm the readiness descriptor's `initial_value` is 0
  (`INVALID == 0`), which the Counter path requires.
- Emit the opaque compile-time and runtime argument blocks to both kernels and
  decode them with `McastArgs`. Chain all following operation arguments from the
  helper's reported next offsets.
- In `reader_final_topk.cpp`, construct the sender face and replace the readiness
  broadcast (`:40`–`:46`) with `send_signal()`.
- In `writer_local_topk.cpp`, construct the receiver face and replace the
  readiness wait (`:56`) with `receive_signal()`.
- Remove the per-round `VALID`/`INVALID` resets **from the readiness channel
  only** — `reader_final_topk.cpp:40` and `writer_local_topk.cpp:101`. The
  coordinator's `sender_sem.set(INVALID)` reset of the arrival counter stays: it
  is operation-owned and is safe where it is, because every round-*j* increment
  has landed by the time `sender_sem.wait(Wt_final)` returns.
- Keep the value/index unicast transfers and the worker-to-coordinator arrival
  counter operation-owned and outside the helper.

### Why first, and what Counter actually buys

Smallest atomic unit: one factory, two kernels, one dense control channel, no
helper API change, and no shared wire with a deferred kernel.

Counter does more than avoid the historical Flag rollback — **it fixes a live
lost-wakeup race in the current code.** The worker clears the readiness flag at
`writer_local_topk.cpp:101`, *after* its `sender_sem.up` and atomic barrier at
`:97`. Once that increment lands, the coordinator's `sender_sem.wait(Wt_final)`
(`reader_final_topk.cpp:52`) can return and re-multicast `VALID` (`:45`) before
the worker executes `:101`, which then erases the new signal. Both sides then
deadlock — the worker at `:56`, the coordinator at `:52`. The window is narrow
(the coordinator must clear `reserve_back` and three semaphore operations before
the worker completes one local L1 write) but it is reachable. A monotone Counter
has no reset and cannot lose the wakeup.

Report this as a bug fix, not only as a migration.

### Validation

1. Run one focused `W=8192` multicore case from
   `tests/ttnn/unit_tests/operations/reduce/test_topk.py` under `--dev` from a
   fresh JIT cache to prove fresh kernel compilation.
2. Run the complete `TOPK-MULTICORE` inventory from `migration/test_map.json`.
3. Run the mcast host tests and complete mcast device-helper suite.
4. Rebuild with `./build_metal.sh` because the host factory changes.

---

## 2. Sharded LayerNorm pre-allgather

### Prerequisite (blocking): split the shared reader compile-time argument builder

`CompileTimeArgs::build` in `sharded_layernorm_factory_helpers.cpp` emits **one**
`args.reader_sender` (`:615`), one `args.reader_receiver_all_to_all` (`:637`) and
one `args.reader_receiver` (`:657`), shared by all three sharded-LayerNorm
variants. Only the kernel *path* branches on `is_pre_all_gather`
(`KernelPaths::get`, `:454`); the argument lists do not.

Splicing a helper block into `args.reader_sender` therefore shifts indices under
the post-allgather and plain-sharded readers, both of which this plan defers.
That is an atomic-unit violation: a shared wire cannot be migrated on one face
and left raw on another.

Resolve it one of two ways before the migration starts:

- **Preferred:** refactor `CompileTimeArgs::build` so each variant emits its own
  reader sender/receiver lists. This is a mechanical host-only change with no
  device-semantics delta, so it can be landed and validated on its own —
  rebuild, then run `LN-PRE-ALLGATHER`, `LN-POST-ALLGATHER` and `LN-SHARDED`
  and require no behavioral change.
- **Alternative:** promote post-allgather and plain sharded LayerNorm into the
  same migration unit and re-plan all six readers together. This contradicts the
  current deferrals for both, and both have open formulation questions
  (see *Deferred* below), so it is the worse option.

This prerequisite — not helper-API distance — is why LayerNorm ranks second.
With the shared builder in place, this unit is structurally *harder* than
Matmul in0, whose factories already branch their in0 argument vectors per path.

### Scope

- `ttnn/cpp/ttnn/operations/normalization/layernorm/device/sharded_layernorm_factory_helpers.cpp`
- `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core_sharded.cpp`
- `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp`
- `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp`

### Migration plan

- Describe each control channel with the helper class that matches its geometry.
  The operation's own "1D" and "2D" vocabulary does not map onto the helper class
  names, so state the class per path explicitly:
  - **whole-grid reduce** (the operation's 1D path, `use_two_stage_reduce=false`):
    one multicast over one rectangle with the sender **inside** it → `Mcast2D`,
    fan-out = area − 1.
  - **two-stage reduce** (the operation's 2D path): a family of row- or
    column-oriented line multicasts → `Mcast1D`.
  Cover both explicitly; do not infer one geometry from the other.
- Adopt or replace the existing reduce semaphore pair
  (`reduce_receiver_semaphore_id`, `reduce_sender_semaphore_id`).
- Configure with `handshake=true` and `DataReadyMode::Flag`.
- Insert complete helper compile-time and runtime argument ranges at a single
  opaque ABI boundary. All following operation arguments must chain from the
  helper's reported next offsets.
- In the sender, replace the raw readiness wait/reset and semaphore multicast
  (`reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp`, the
  `set(VALID)` / `wait(num_blocks - 1)` / `set(0)` / `set_multicast` sequence)
  with the v10 handshaked `send_signal()` path.
- In the receiver, replace clear/up/wait with `receive_signal()`.
- Keep gather reads, CB ownership, the second-stage reduce semaphore
  (`reduce_second_stage_semaphore_id`), and the final atomic barrier
  operation-owned.
- Add host geometry tests for row-wise, column-wise, offset, one-stage, and
  two-stage layouts. Confirm dense/fan-out assumptions rather than inheriting
  the raw factory's counts.

### Why second

API-003 is implemented at v10, so the acknowledged signal-only behavior that
caused this pair's rollback now exists in the helper, and the historical partial
migration passed 32/32 tests. The remaining cost is the shared-builder
prerequisite above plus the two-stage geometry coverage.

### Validation

1. Land and validate the prerequisite builder split on its own (rebuild, then
   `LN-PRE-ALLGATHER` + `LN-POST-ALLGATHER` + `LN-SHARDED`, no behavior change).
2. Run one 8x4 pre-allgather parametrization under `--dev` from a fresh JIT cache.
3. Run the complete `LN-PRE-ALLGATHER` inventory from
   `tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py`.
4. Cover LayerNorm and RMSNorm, whole-grid and two-stage reduce, offset grids,
   and non-tile-aligned widths.
5. Re-run `LN-POST-ALLGATHER` and `LN-SHARDED` to prove the deferred faces are
   unaffected by the wire change.
6. Run the mcast host/device helper suites.
7. Rebuild with `./build_metal.sh` because the host factory changes.

---

## 3. Matmul in0 multicast

### Prerequisite (blocking): decide how batch validity is expressed

The in0 channel carries two distinct exchanges on **one** semaphore cell:

- the normal in0 block transfer (a plain data-ready flag), and
- a sparsity batch-validity exchange that encodes three states in the *value* of
  that same cell — `INVALID`(0) / `VALID`(1) / `IGNORE_BATCH`(2)
  (`reader_bmm_tile_layout_in0_sender_padding.cpp:190`–`:202`; the receiver reads
  the landed value after `wait_min(VALID)` in
  `reader_bmm_tile_layout_in0_receiver.cpp:55`–`:57`).

**The previously proposed one-word payload channel is withdrawn.** Multicasting a
`0`/`1` word into a dedicated scratch CB was rejected for three reasons:

- **It regresses the path it targets.** Today a skipped batch costs exactly one
  multicast flag write — the sender `continue`s before any data multicast, so the
  control exchange is the *only* traffic for an invalid batch. Replacing it with
  `send()` adds a data multicast, a second signal multicast and a fence, roughly
  doubling the cost of precisely the case sparsity exists to make cheap.
- **It needs a new stable CB in matmul's L1 budget**, identical across all
  participants — the most L1-pressured op in the tree, and a question the
  original plan itself left open ("stop and reassess").
- **It is a protocol redesign inside a rollout.** `apply-dm-helper` consumes a
  proven helper; it does not design one.

Pick one of these instead, before the migration starts:

- **Preferred — file API-007 through `tune-dm-helper`: let the data-ready Flag
  carry a small caller-supplied value.** `receive_signal()` already returns
  `uint32_t` and already clears the flag; the change is `send_signal(uint32_t
  value = VALID)` plus a `wait_min` + return-the-observed-value on the receive
  side. It is wire-compatible, adds zero packets, expresses the existing
  behavior exactly, and keeps the value/flag distinction inside the helper where
  it can be documented and tested. Bumps `MCAST_PIPE_API_VERSION`, so the
  existing 13 migrated kernels become stale and are remigrated as Tier 0 —
  budget for that.
- **Alternative — migrate the in0 data channel only** and leave the sparsity
  control exchange operation-owned, exactly as TopK's arrival counter and
  LayerNorm's second-stage semaphore stay operation-owned. This needs a hazard
  ruling first, because both uses share one semaphore cell: the helper's
  `ReceiverPipe` constructor sets that cell to `INVALID` and `send()` re-asserts
  `VALID` before each broadcast, and those must be proven compatible with a raw
  typed-control round interleaved between data rounds. Record the ruling in
  `hazards_catalog.md`.

Do not start item 3 until one of these is chosen and, for the first option,
materialized and unit-tested upstream.

### Scope

- `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp`
- `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp`
- `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`

### Migration plan

- Reuse the host geometry already established for Matmul in1 to emit an in0
  `Mcast1D` or `Mcast2D` block for every legacy, descriptor, and sparse binding.
  Note the in1 prior art covers only the 1D and 2D factories; the **sparse
  factory has no migrated in1 binding**, so its in0 geometry is built from
  scratch. Budget for that and map it per 0.3.
- **Emit the helper block inside the interleaved branch only.** Both in0 sender
  kernels are bound from the same descriptor variable, but the compile-time
  vectors already branch on `in0_block_sharded`
  (`..._2d_program_factory.cpp:402` vs `:431`;
  `..._1d_program_factory.cpp:360` vs `:390`), so the deferred block-sharded
  kernel keeps its own layout as long as the block goes into the `else` branch.
- **Do not splice the block at the front of the vector.** The block-sharded paths
  mutate indices `[0]` and `[1]` by position
  (`..._1d_program_factory.cpp:615`–`:616`, `:634`–`:635`;
  `..._2d_program_factory.cpp:2249`–`:2250`). Append at a known offset and chain
  the trailing `TensorAccessorArgs` blocks — two of them on the in0 sender
  (`..._2d_program_factory.cpp:470`–`:471`), plus `num_batch_compute` — from the
  helper's reported next offsets.
- Decode the host block through `McastArgs` and migrate the normal in0 block
  transfer to `send()`/`receive()`.
- Preserve the divergent active-ack count on uneven grids rather than deriving
  it from rectangle area.
- Express batch validity per the prerequisite decision above.
- Preserve `SKIP_MCAST`, in0-sharded extraction, fused-op parsing, and the
  on-device `nnz` contract checks.

### Why third

Broadest host surface of the three, an unmapped sparse binding, and a blocking
formulation decision on batch validity. Both kernels have passed earlier helper
migrations and the 1D/2D factories already integrate the host helpers for in1,
which is real prior art — but it does not extend to the sparse factory.

### Validation

1. Start with one 1D interleaved Matmul parametrization under `--dev` from a
   fresh JIT cache.
2. Run one sparse Matmul case without `nnz` to exercise the batch-validity
   exchange before running dense coverage.
3. Run sparse Matmul with and without `nnz` from
   `tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py`, via the
   `MM-SPARSE-IN0` inventory added in 0.3.
4. Run the complete `MM-IN0-INTERLEAVED` inventory, including 1D/2D,
   legacy/descriptor, tiny-tile, subdevice, uneven-width, padded, and fused
   paths.
5. Run `MM-BLOCK-SHARDED-HYBRID` to prove the deferred block-sharded in0 face is
   unaffected by the argument-vector change.
6. Measure and record the perf delta on the sparse path specifically — a skipped
   batch must not cost more multicast traffic after the migration than before.
7. Run the mcast host/device helper suites and opaque-ABI source audit.
8. Rebuild with `./build_metal.sh` because multiple host factories change.

---

## Deferred until after these three

- **LayerNorm post-allgather:** still needs a deliberate formulation for
  source loopback when the sender is outside the receiver rectangle, plus
  confirmation of ragged-grid fan-out semantics. Note it shares the reader
  compile-time argument builder with item 2 — see that item's prerequisite.
- **Plain sharded LayerNorm:** combines acknowledged signal-only traffic with
  one-gate/multi-block streaming and reused semaphore state. Shares the same
  builder.
- **Block-sharded Matmul in0:** retains independent data/signal loopback modes,
  rotating roles, and sender-side ready participation. Its compile-time vector
  is already branch-separate from interleaved in0, so item 3 does not move its
  wire — but item 3's validation must prove that.

## Execution rules

- Treat each numbered item as an atomic host-plus-kernel migration unit; do not
  leave one face raw and the other helper-backed. A shared argument builder makes
  every kernel reading it part of the unit — resolve that before migrating, not
  during.
- Run `apply-dm-helper` with `--mode=halt`.
- For each item, rebuild host code before device validation.
- Activate `/localdev/sjovic/tt-metal/python_env/bin/activate` and run device
  tests through `scripts/run_safe_pytest.sh`.
- Run device tests sequentially. First prove one exact parametrization compiles
  and passes from a fresh JIT cache, then run the complete mapped inventory.
- Use `--dev` for the first fresh-compilation case.
- Commit each unit atomically and write back its ledger entries at the current
  `MCAST_PIPE_API_VERSION` **in the same step as the commit**.
- Update the census, kernel annotations, ledger Markdown mirror, test map,
  migration report, and dashboard after each unit's implementation and mapped
  validation are complete — per unit, not once at the end of the sequence.
