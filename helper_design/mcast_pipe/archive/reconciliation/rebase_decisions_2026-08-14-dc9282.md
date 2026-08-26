# Automated rebase decisions — `sjovic/mcast-migration` — 2026-08-14

## Result

The 58-commit branch was rebased from old helper-library baseline
`4a1d6a97ca9bd4efabd0ad6115fcb30538851c90` onto fetched baseline
`dc9282be7d5e9d5a4b9137c1bf327de8d923e18e`.

- Pre-rebase HEAD: `40692d1e9ee4beb2a2db4196e1718407bf254bc4`.
- Rebased HEAD: `91bf395736241f7b62941c19383028e4f53da2ad`.
- Backup: `backup/mcast-migration-prerebase-20260814-40692d1`.
- Range-diff: all 58 commits mapped in order; 39 patch-identical and 19 changed by upstream/conflict
  composition; no commit was added or dropped.
- Publication: nothing was pushed.

## Repository and fetch decisions

The remote helper-library history had been rewritten, so an ordinary fetch rejected the tracking-ref
update as non-fast-forward. I confirmed the advertised remote SHA with `git ls-remote`, then
force-updated only `refs/remotes/origin/llk_helper_library` to the advertised
`dc9282be7d5e9d5a4b9137c1bf327de8d923e18e`. No branch or tag was published.

The pre-existing `tt_metal/third_party/tt_ops_code_gen` checkout was detached temporarily at the
parent-recorded commit so the rebase could run from a controlled state. It is restored after the
workflow to the user's original `codex/share-helper-skills` branch at `7e974cd3`. The UMD submodule is
left at the new rebased parent's recorded commit `2f923261d5a383cacaa22b2c41a75612c12cf344`.

The parent branch records `tt_ops_code_gen` at `4860704b`, exactly as the pre-rebase branch did, while
the restored checkout is at the squashed replacement `7e974cd3`. No remote ref contains `4860704b`;
the local backup branch retains the object, so the state is usable locally but a fresh clone or CI
submodule update cannot reproduce it. This inherited publication blocker was not created by the
rebase and must be resolved before the parent branch is published.

## Conflict and semantic decisions

### Helper and initial migration composition

- Preserved the new baseline's `DataflowBuffer` conversions and runtime layouts.
- Kept helper ownership where the final topology is expressible.
- Matmul in1 retained receiver semantic IDs 4/5 and sender IDs 10/11, with the lifecycle
  reserve → helper receive/send → push.
- GroupNorm retained remote source-read acknowledgement counters outside the no-handshake helper
  channels.
- `mcast_pipe.inl` retained the upstream loopback-aware source-lifetime fence.
- GroupNorm sender retained upstream fp32 zero-fill/data-width behavior while keeping the helper wire.

### LayerNorm post-allgather rollback

The older helper fix was already upstream. I kept the post-allgather sender/receiver raw because one
host route uses INCLUDE-source delivery while the sender can sit outside the receiver rectangle, a
topology the helper cannot express without changing operation behavior. Both final files are
byte-identical to the new baseline, and the stronger upstream tests remain.

### Conv host geometry

- Height/default weights use one host `Mcast2D`, preferred writer NOC, and helper arguments.
- Block-sharded weights keep their separate fixed-line ABI.
- Upstream's removal of `partials_cb_uses_output` from the compute-kernel arguments is preserved. The
  conflict-composed factory retains only a dead diagnostic local and `log_debug`; it does not affect
  behavior and can be removed separately.
- The width-sharded activation helper path was not part of an unresolved source conflict; its factory
  composition preserved the branch's established route.
- The block-sharded activation-helper experiment and its performance chain remain reverted as the
  branch intended. The raw activation path keeps the upstream
  `smuggled-rta-ok: compile-time workload-owned buffer` annotation.

### GroupNorm geometry and ABI

- Kept three fixed multicast blocks: middle, first edge, and last edge.
- A missing edge remains a sender-only degenerate block rather than changing the wire shape.
- Preserved `DataflowBuffer` use and the remote source-read acknowledgement gate.
- The fp32 flag remains after helper arguments: Welford receiver offset +9 and sender offset +11 after
  the opaque-ABI commit.

### Sort control protocol

- Preserved the upstream partial-grid hang fix: each helper multicast uses the bounding box for
  fan-out while `num_active = core_range.num_cores()` supplies the smaller active acknowledgement
  count.
- Preserved UInt16 reader fields: runtime base +5, stage CB +6, helper +7.
- Split the previously ambiguous control channel into a handshaked row-start Counter and a
  no-handshake sub-stage Counter; writer-done remains the operation-owned raw semaphore ID 3.

### Source lifetime

Preserved the branch's `SourceL1Guard` performance/correctness semantics: loopback delivery is always
ACK-fenced; remote-only sends may omit the source guard when the source lifetime proves safe. Flag and
Counter ownership rules remain unchanged.

### TopK readiness

Preserved upstream `DataflowBuffer` use. The helper owns only monotone readiness Counter signaling;
the operation-owned arrival counter and value/index fan-in remain unchanged.

### LayerNorm pre-allgather arguments

Kept helper runtime arguments as a prefix when present, then retained the upstream vector-capacity
optimization by reserving `args.size() + operation_argument_count`.

### Matmul readers and factories

- Preserved `DataflowBuffer` conversions together with helper ownership.
- Preserved upstream global-circular-buffer receiver validation.
- Narrowed the rotating multicast rectangle to `all_cores_with_work`; shard cores outside that
  receiver rectangle skip the receive face through `core_in_receiver_rect()`.
- Preserved DRAM-aligned stride logic/comments, UInt/padding behavior, and the later rename from
  `in1_pipe` to `weights_bias_pipe`.
- Independent review found a separate, pre-existing risk in the Matmul in0 block-sharded rotating
  path: a partially filled final grid row can make the dense receiver bounding box larger than the
  active kernel set, while the helper defaults its acknowledgement count to fan-out. The same helper
  resolution is present in the pre-rebase branch, so this is not a rebase regression, but it must be
  fixed or guarded before the branch is treated as generally production-safe.

## Verification

- Rebased baseline is an ancestor of the new HEAD; exactly 58 commits are above it.
- `git diff --check` passed.
- Full Release `./build_metal.sh` passed.
- Sort exact compile-focused test passed, followed by six expanded UInt16/handshake cases.
- Helper device and source-audit suites passed 97/97 under `--dev`.
- Host geometry suite passed 32/32.
- Focused Matmul, TopK, GroupNorm Welford, and LayerNorm pre-allgather probes passed 4/4.
- Complete post-rebase operation coverage for the 12 rebase-touched kernels passed with 1,228 tests,
  190 expected skips, and 90 expected xfails; there were no failures or hangs.

The stored Matmul node ID had gained an upstream `silicon_arch_name=blackhole-mesh_device=(1, 1)`
parameter prefix. The first invocation using the stale ID selected no test; rerunning the current
collected ID passed.

## Rollout reconciliation

All 91 ledger paths exist, all 17 migrated kernels still use the helper, and no entry was removed,
renamed, or clobbered. Twelve migrated kernels were rebase-touched and carry `needs_recheck`; the exact
recognition-family delta produced no new in-scope multicast-pipe candidate. Details are in
`reconcile_2026-08-14-rebase-dc9282.md`.

## Independent consensus audit

Claude and the primary reviewer independently reconstructed the rewrite, checked every changed branch
delta, and reviewed the behavior-sensitive producer/consumer, CB, semaphore, NoC, and helper contracts.
They agreed that the rebase is faithful and safe to keep locally: no code defect was introduced by the
rebase, and no upstream behavior was lost. The artifact clarifications in this document do not change
that verdict.

They also agreed that the inherited Matmul partial-grid acknowledgement issue is a separate branch bug
and that the inherited unreachable `tt_ops_code_gen` gitlink blocks publication. Historical
`verified_at_commit` fields remain unchanged because they identify the tree on which their recorded
tests actually ran; the deferred helper-neutral Sort writer does not receive `needs_recheck`, which is
defined only for migrated entries. Nothing was pushed.
