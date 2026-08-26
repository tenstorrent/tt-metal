# Archived: Matmul migration feedback remediation plan — 2026-08-07

Working branch: `sjovic/mcast-migration` at `97dde16f5e1` when this plan was
written.

Intended branch baseline: `llk_helper_library` at `4a1d6a97ca9`, confirmed as
an ancestor of the working branch. The direct pre-Matmul-migration comparison
point is `15eb8aec4fa`, the parent of `97dde16f5e1`.

This work will not rebase, reset, push, or use a worktree. Existing unrelated
working-tree changes and untracked files are not part of the remediation.

## Outcome and acceptance criteria

The final change must satisfy every item in `matmul_feedback.md` and resolve
API-009 without changing the original Matmul protocol:

- the host helper represents rotating senders independently of the receiver
  rectangle;
- block-sharded Matmul uses the original output-work receiver rectangles and
  leaves shard-only cores sender-only;
- degenerate self-only sends preserve the original local `noc.async_write`
  behavior and do not introduce an immediate barrier;
- all Matmul bindings use `McastArgs` directly, with no `MCAST_ARGS` define or
  conditional legacy multicast ABI;
- multicast objects use the same `in0_noc` / `in1_noc` values as their kernel
  descriptors;
- new multicast CT and RT blocks are appended at a clear ABI boundary and all
  downstream indices are derived from `next_compile_time_args_offset()` and
  `next_runtime_args_offset()`;
- unrelated comments are restored exactly;
- correctness matches the recorded inventory and performance is no more than
  1.5% slower than the matched pre-migration baseline on the same Blackhole
  device/AICLK, using three warmups and 20 measured records.

## Design decision for API-009

The required behavior is already representable by the device wire. A rotating
RT block contains the destination rectangle followed by an ordered sender
coordinate list. `SenderPipe` also determines whether the current sender is in
that rectangle and derives the exclude-source fan-out. Therefore API-009 is a
host-model limitation, not initially a device-wire limitation.

Extend the rotating host construction so the receiver rectangle and ordered
sender set are separate inputs. For `Mcast1D`, preserve the line-family model
while allowing each line's ordered sender sequence to extend outside its
receiver line. For `Mcast2D`, allow one explicit ordered sender sequence for the
fixed rectangle. The participating semaphore set becomes the union of the
receiver rectangle and all senders.

For a dense mixed inside/outside sender sequence, emit the existing
`ACK_EQUALS_FANOUT` sentinel so each `SenderPipe` derives its own correct ACK
count: rectangle area minus one for an inside sender, rectangle area for an
outside sender. Preserve explicit numeric `num_active` only for protocols with
a genuinely divergent acknowledging subset. This should keep
`MCAST_PIPE_API_VERSION` at 11; bump it only if a focused host/device test proves
that the existing wire cannot encode the contract.

Reject malformed host configurations early: empty sender sequences, duplicate
senders, inconsistent 1D line counts/order, senders outside the supplied
participating topology, and an explicit divergent ACK count that cannot be
valid for every sender role.

## Execution order

### 1. Freeze evidence and add regression tests first

- Save the current diff and record the exact hashes above.
- Preserve the existing `generated/mcast_migration_rt` artifacts. Use the
  `4a1d6a97ca9_*` and `gate7_20260805_*` Matmul records as historical baselines,
  and use `15eb8aec4fa` as the source-level pre-migration reference.
- Add host tests for an unchanged receiver rectangle with an ordered sender set
  containing both inside and outside senders. Assert the CT rotating count/ACK
  policy, RT rectangle, sender order, role queries, and owned-semaphore union
  for both NoCs, both 1D orientations, offset rectangles, and 2D.
- Add a device helper case that executes at least two rounds across the
  inside/outside boundary and checks payload plus handshake completion. Add a
  1x1 degenerate case whose output proves the helper-owned local copy works.
- Extend the source audit to reject `MCAST_ARGS` in the migrated Matmul kernels,
  hard-coded post-helper ABI indices, recomputed NoCs at multicast construction,
  and receiver-grid expansion to `all_cores`.

These tests should fail for the current API/topology before implementation and
provide a narrow compile/debug loop.

### 2. Restore the helper's degenerate local-copy semantics

- Change `SenderPipe::local_copy_` from `noc.async_read` plus an immediate read
  barrier to the original same-core `noc.async_write` form.
- Do not add an immediate write barrier. Preserve the caller's original
  synchronization/lifetime behavior; Matmul already reaches its existing final
  write barrier.
- Run the focused 1x1 helper test, then the complete helper device suite in
  `--dev` and normal mode. This helper change precedes removal of any Matmul
  call-site fallback.

### 3. Implement API-009 in the host helper

- Add the independent rotating-sender representation to `Mcast1D` and
  `Mcast2D`, while leaving fixed-sender and current rotating constructors
  behaviorally unchanged.
- Generate the true receiver rectangle for every sender, the ordered sender
  coordinate list for every participant, and semaphores on receiver union
  sender cores.
- Use the existing dense-fan-out sentinel for mixed inside/outside senders; do
  not add a caller-supplied device behavior knob.
- Update Doxygen/wire comments and mark API-009 Implemented with the evidence.
  Record the decision in `changelog.md`. Do not claim or introduce a wire
  version change unless tests force one.
- Rebuild host code with `./build_metal.sh`, then run all
  `McastHostFixture.*` tests and the complete helper device suite sequentially.

### 4. Restore Matmul topology before simplifying the kernel

- In both legacy and descriptor builders in the 1D and 2D factories, reconstruct
  the exact pre-migration block-sharded receiver rectangles from
  `15eb8aec4fa`. Pass the original ordered shard sender sequence to the new host
  API instead of widening the receiver grid.
- Restore the distinction between working receivers, non-working receivers
  inside the original rectangle, and shard-only senders outside it wherever the
  factory/kernel partition still requires it.
- Verify emitted RT args directly in host-level tests or a focused factory
  probe: an outside sender must receive the original rectangle, must not be a
  receiver, and must wait for all receivers; an inside sender must exclude self
  and wait for the remaining receivers.
- Only after that geometry is green, delete the
  `else if constexpr (!extract_shard_sub_blocks)` local-copy branch in
  `reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` and
  always let `SenderPipe::send()` own the degenerate case.

### 5. Clean the Matmul ABIs and migration-only diff

- Move `in0_noc` and `in1_noc` selection before multicast construction and pass
  those exact variables to the helper objects and kernel descriptors in every
  legacy/descriptor builder. Apply the same rule to the sparse factory.
- Give every binding of the shared in0/in1 sender kernels a real `McastArgs`
  block, using an inactive block when multicast is disabled. Remove the
  `MCAST_ARGS` defines and all `#ifdef MCAST_ARGS` / `#ifndef MCAST_ARGS`
  branches from the kernels.
- Re-layout CT and RT arguments so operation-specific legacy fields remain
  together and the opaque multicast block is appended at the clean final ABI
  boundary (before TensorAccessor blocks only when those must remain last).
  Update host producers and kernel consumers together.
- Chain anything after the multicast block exclusively through
  `next_compile_time_args_offset()` / `next_runtime_args_offset()`. Re-audit
  TensorAccessor bases, sparsity fields, batch/fused-op tails, optional bias and
  output-sharded fields, and every program-cache override/descriptor patch.
- Restore the pre-migration in0-sender padding comments byte-for-byte and avoid
  unrelated cleanup.
- Run formatting only on touched files and inspect the final diff against
  `15eb8aec4fa` specifically for receiver geometry, transfer primitive,
  barriers, argument order, NoC selection, defines, and comments.

### 6. Correctness validation, sequentially

All device tests run after activating
`/localdev/sjovic/tt-metal/python_env/bin/activate` and only through
`scripts/run_safe_pytest.sh`. No tests run in parallel.

1. From an empty isolated JIT cache, run one focused `--dev` parametrization
   that compiles the block-sharded 2D path with a sender outside the receiver
   rectangle. The `-k` expression will filter on a value token and contain no
   `=` character.
2. Run the same focused case normally to expose timing-sensitive semaphore or
   CB races.
3. Run focused 1D interleaved, 2D interleaved, 2D transposed block-sharded,
   1x1 degenerate, padding, fused-bias/activation, and sparse cases, covering
   both legacy `Program` and `ProgramDescriptor` construction.
4. Run the full `MM-IN0-INTERLEAVED`, `MM-BLOCK-SHARDED-HYBRID`,
   `MM-IN1-RECEIVER-2D`, `MM-IN1-ALL`, and `MM-SPARSE-IN0` inventories from
   `migration/test_map.json`. The `MM-IN1-ALL` acceptance baseline remains 302
   passed and 188 expected skips unless upstream test inventory has changed;
   any delta is investigated, not normalized away.
5. Run program-cache reuse coverage to validate every runtime-argument override
   index after the ABI move.
6. Re-run the full helper suite in normal mode after production integration.

If a test hangs, first use the safe runner's triage output and inspect the first
core/RISC stack. Re-run with `--dev`/Watcher when appropriate, but do not combine
Watcher and tt-triage. Reset with `tt-smi -r` only after a confirmed hang.
The ops-codegen TTNN implementer prompt may be used in standalone-debug mode for
a failing kernel; its scope is diagnosis/fix, while this plan's migration
guardrails and original protocol remain binding.

### 7. Performance validation

- Run the checked-in real-time-profiler Matmul harness sequentially for:
  `matmul_2d_sdxl_ff_gelu`, `matmul_1d_sdxl_resnet_960_320`, and
  `matmul_2d_transpose_mcast`.
- Add one production-reachable 1x1/degenerate Matmul case and one
  block-sharded case where the sender span exceeds the receiver span, because
  those are the two feedback-sensitive paths not isolated by the existing
  three cases. Confirm the intended kernels compiled before crediting a result.
- Use AICLK 800, three warmups, 20 measured records, and compare median
  `Kernel duration (ns)` against matched pre-migration records on the same
  machine. Repeat any result near or above the 1.5% limit before diagnosis.
- If a case regresses by more than 1.5%, isolate in this order: ABI-only cleanup,
  helper local-copy path, receiver geometry/ACK fan-out, then active send hot
  path. Do not trade correctness or restore the widened rectangle to recover
  performance.
- Run `python -m tracy -r -m pytest ...` only for a focused regression after the
  safe correctness run is green, and use the generated CSV's
  `Kernel duration (ns)` field to localize it.

## Documentation and completion

After all gates pass, update API-009, `changelog.md`, the Matmul migration logs,
ledger/test-map coverage, and the dashboard inputs with exact commands, pass
counts, JIT evidence, performance artifacts, medians, and deltas. Keep source
changes and evidence updates in reviewable commits. Do not push without explicit
permission.

## Known constraints, not blockers

- The checkout currently contains unrelated modified/untracked files; they will
  be preserved and excluded from commits.
- Historical performance artifacts exist for the intended baseline. If a
  matched baseline must be rerun, do it sequentially in this checkout without a
  worktree and without reset/rebase; otherwise retain the recorded baseline and
  clearly label the comparison.
- Kernel changes require no rebuild, but the host helper/factory changes do, so
  `./build_metal.sh` is mandatory before the full validation gates.

## Execution result — 2026-08-07

All implementation and validation gates above completed. API-009 was resolved
on the existing v11 wire; receiver geometry, NoC selection, unconditional
helper ABIs, fixed-operation argument boundaries, original comments, and the
degenerate write primitive are protected by host/device tests and a 17-test
source audit. The host build passed, helper normal and Watcher suites passed
80/80 each, host wire tests passed 30/30, full Matmul passed 816 with 310
expected skips and 2 known xfails, and sparse Matmul passed 18/18.

At 800 MHz, the matched 3-warmup/20-record medians versus `4a1d6a97ca9` were
+0.643%, +0.809%, and -0.045% for the three established Matmul cases. New
current-snapshot records also cover the 1x1 degenerate path (2,548.925 ns) and
sender-span-greater-than-receiver-span path (11,787.407 ns), with kernel-source
assertions. Historical matched baseline artifacts do not exist for those two
new cases, so only their current absolute measurements—not regression deltas—
are reported. No checkout mutation was used to manufacture a baseline.
