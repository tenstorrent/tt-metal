# mcast `Pipe` — change log

Round-by-round record of the helper's API and rollout. Each round: trigger →
decisions → artifacts touched → device verification. Diff API states here when a new
feedback round lands.

---

## Round 34 — one template-owned runtime base (2026-08-23)

- Bumped the helper to API v14 and removed `McastArgs`' dynamic runtime-base
  constructor, mutable stored base, and duplicate instance base/end accessors.
  `RT_BASE` is now the only source of truth for every helper runtime read and
  chained boundary.
- Refined the runtime ABI guardrail for operation data whose length is known
  only at runtime: fixed operation prefix, complete helper block, then variable
  operation tail. Compile-time ABIs remain operation-first/helper-last, and
  fixed-width runtime ABIs retain the ordinary helper-tail layout.
- Applied that order to all five previous dynamic-base consumers: four Matmul
  layouts and the Conv3D writer, including both Matmul legacy and descriptor
  producers. The Matmul in1 width-sharded parser now advances across the full
  variable DRAM bank/stride payload before decoding any fused-operation tail.
- Validation passed: release host build, 36/36 host helper gtests, 80/80 helper
  device tests under `--dev`, 27/27 source audits, exact 1D, block-sharded 2D,
  and DRAM-width-sharded in1 Matmul gates, and the exact Conv3D gate. Conv3D's
  known Watcher skip (#37184) passed unchanged without Watcher. The existing 31 migrated kernels and 27
  migrated bindings are stamped v14; pending and deferred inventory is
  unchanged.

---

## Round 33 — tagged optional helper ABI and final feedback cleanup (2026-08-22)

- Bumped the helper to API v13. Every present multicast compile-time block now
  starts with a true presence tag and contains seven words total; an absent
  block contains only a false tag and no runtime payload.
- Folded optionality into `McastArgs` with compile-time present/absent
  specializations. The absent form reads no payload, advances the actual encoded
  offsets, and compile-time rejects sender or receiver construction.
  `OptionalMcastArgs` and operation-owned Matmul presence flags were removed.
- Applied tagged serialization to the complete migrated fleet and replaced the
  five inactive Matmul/Sparse shared-kernel bindings with the opaque absent
  helper block. Present, absent, and chained Matmul routes passed under Watcher.
- Moved identical in0 helper tails outside the sharded/interleaved conditionals
  in both 1D Matmul builders. The cross-operation audit found and fixed the same
  duplication in both 2D builders; a source test now enforces all four sites.
- Validation passed: release host build, 36/36 host helper gtests, 80/80 helper
  device tests under `--dev`, 26/26 source audits, asymmetric 1D Matmul, Sparse
  Matmul, and 2D Matmul focused gates. The existing 31 migrated kernels and 27
  migrated bindings are stamped v13; pending and deferred inventory is unchanged.

---

## Round 32 — migration feedback and API-v12 fleet write-back (2026-08-22)

- Normalized every migrated kernel ABI to operation-owned compile/runtime
  prefixes followed by opaque multicast tails. Added a dynamic runtime base and
  `OptionalMcastArgs` so variable and inactive paths do not split the ABI or emit
  synthetic helper blocks. The v12 wire format remains self-describing.
- Added `Mcast1D`/`Mcast2D` append-style host APIs and converted every migrated
  producer, including legacy, descriptor, cache-reuse, optional, and multiple-pipe
  paths. A host gtest verifies exact equivalence with the getter form.
- Added concrete `McastArgs::SenderPipe`/`ReceiverPipe` aliases and converted all
  mixed-role storage. This exposed and fixed Group Attention Matmul constructing
  a receiver on sender-only cores; role asserts remain enabled.
- Preserved explicit divergent ACK-count overrides where partial bounding boxes
  contain inactive landing cores. The review suggestion to derive a dense count
  was corrected after the exact uneven-width route reproducibly hung without
  the override and passed after restoration.
- Validation passed: release host build, focused host gtest, 80/80 helper device
  tests under `--dev`, 24/24 source audits, and sequential focused gates spanning
  all migrated operation families. Three exact routes hit the known unrelated
  Watcher C++17 `ASSERT` compilation incompatibility in other kernels and passed
  unchanged through the safe wrapper without `--dev`.
- The existing 31 migrated kernel rows and 27 migrated host bindings are written
  back at API v12. The 2 pending kernels, 5 pending bindings, and 71 deferred
  kernels retain their prior dispositions.

---

## Round 31 — Tier 2.10 post-allgather LayerNorm (2026-08-16)

- Production commit `6cc49825476` migrates the post-allgather sender/receiver pair at API v11. Dense
  one-dimensional multicast uses helper loopback; a non-1D sender keeps its local CB21-to-CB15 copy
  operation-owned while an independent per-line `Mcast2D` serves the remote rectangle.
- Generalized the shared LayerNorm multicast descriptor across pre- and post-allgather without changing
  the ordinary sharded path. Added offset-dense and outside-sender row geometry host coverage.
- Release build, exact fresh JIT, all post/pre/plain LayerNorm inventories, 34 host tests, 80 helper
  tests, and the source audit passed. Every production file shrank independently.
- Matched 800 MHz medians improved from 4009 to 3880 ns for LayerNorm and 4020 to 3617.5 ns for RMSNorm.
  API expansion: NO. Current rollout state: 23 migrated, 2 pending, and 79 deferred kernel rows; 21
  migrated and 5 pending host bindings.

---

## Round 30 — approved Tier 0/1/2 rollout completion (2026-08-16)

- Reconciled the approved plan inventory to 104 kernel paths, advanced the existing migrated fleet to
  API v11 after complete verification, and retained the two interleaved Matmul paths as pending because
  their historical matched-performance checkout was not separately authorized.
- Completed Tier 0.2 block-sharded Matmul, Tier 1.6 DeepSeek sampling, Tier 2.8 rotating group-attention
  Matmul, and Tier 2.9 Conv3D weight multicast. Tier 2.7 DRAM-sharded Matmul remains deferred because
  preserving its forced self-inclusion/exclusion combinations would require an unauthorized API change.
- Conv3D commit `a290ce20281` replaces only the rectangle weight multicast with fixed-sender `Mcast2D`
  instances. Chain and Disabled modes remain unchanged. Build, fresh-JIT, focused and complete Conv3D
  correctness, 32 host tests, 80 Watcher device tests, 18 source audits, and two matched 800 MHz
  performance routes passed; non-grouped and grouped medians improved 0.815% and 0.298%.
- The helper API remains v11. Rollout state after the initial scope: 21 migrated, 2 pending, and 81 deferred kernel rows;
  21 migrated and 5 pending host bindings.

## Round 29 — revert experimental Conv streaming send (2026-08-14)

- Reverted the complete experimental chain with commits `f003c5c3687`,
  `12d2548d4e1`, `d1277247931`, and `9d870bf2da9`. This removes
  `SenderPipe::send_from_cb`, restores block-sharded Conv activation's raw
  producer-overlapped multicast, and restores the established width-sharded
  path and gate.
- The five affected source files are byte-identical to the parent of
  `f3361f57596`. The helper remains API v11 because the experiment had not
  changed `MCAST_PIPE_API_VERSION`.
- Restored the block-sharded Conv activation ledger row to deferred R4,
  removed its no-longer-valid pending host binding, and cleared the
  width-sharded kernel's experiment-only `needs_recheck` flag. The attempted
  migration log is archived as historical evidence.
- Validation passed: `./build_metal.sh`; one exact block-sharded Conv node under
  `--dev`; complete block- and width-sharded feature inventories at 48 passed /
  16 expected skips each; all 80 helper device/wire tests under `--dev`; and all
  32 `McastHostFixture` tests.

## Round 28 — single ledger inventory and design-evidence layout (2026-08-14)

- Folded the separate 91-path text inventory into `migration/ledger.json`. Each
  entry now carries both durable discovery fields and mutable rollout state;
  the former path set matched the ledger exactly before deletion.
- Moved contracts, hazards, feasibility analysis, and bake-off evidence under
  `design/`; kept `proposed_helpers.md` at the top level as the active contract.
- Updated tune/apply/reconcile workflow skills so future rollouts create and
  maintain one ledger rather than parallel inventory files.
- Removed the paused generated `migration/tiers.md` and `migration/report.md`;
  they will be regenerated by the next apply run. Archived the latest static
  reconciliation report with the earlier dated audits.
- Documentation and workflow only: no helper API, production code, migration
  status, build result, or device result changed.

## Round 27 — reconciliation and documentation handoff (2026-08-14)

- Reconciled all 91 census paths against the current source tree: 17 migrated,
  4 pending, and 70 deferred kernels; no missing, renamed, or clobbered paths.
- Expanded the host-binding inventory to include the four block-sharded Matmul
  legacy/descriptor routes and the block-sharded Conv2D activation route. The
  ledger now records 14 migrated-at-v10 and 10 pending bindings.
- Intake only: the current host build passed, `McastHostFixture.*` passed 32/32,
  and the complete helper device/wire suite passed 80/80 under `--dev`. No
  per-operation apply validation or v11 ledger write-back was claimed.
- Consolidated the live handoff around README, ledger, test map, and changelog;
  moved completed plans and superseded reports under `archive/`.

## Round 26 — width-sharded Conv streaming overlap (2026-08-12 to 2026-08-14)

- `8ae4604379e` overlaps tall width-sharded Conv multicast with upstream CB
  production instead of waiting for the entire source block.
- `9686814ea22` gates that streaming path on measured wins, retaining the simpler
  path where overlap does not justify its cost.
- The helper API remains v11. The migrated width-sharded kernel is flagged
  `needs_recheck` until its mapped operation inventory is written back.

## Round 25 — block-sharded Conv streaming integration (2026-08-11)

- `f3361f57596` adds `SenderPipe::send_from_cb` and migrates the production
  block-sharded Conv2D activation multicast while preserving per-burst CB
  readiness and producer/NoC overlap.
- `ccd7b597e92` leaves CB data-type resolution at the call site. This avoids
  embedding operation-specific CB interpretation in the helper.
- The API remains v11. Kernel and host binding are source-integrated but remain
  pending until the mapped Conv validation and performance gate complete.

## Round 24 — Matmul host topology and naming follow-up (2026-08-08)

- `5396be87ecc` names the Matmul weights and bias multicast pipe consistently in
  both migrated in1 kernels; no protocol behavior changes.
- `233f43c7d44` moves block-sharded Matmul's independent ordered sender topology
  into the host helper and removes duplicated factory geometry. Host and source
  audit coverage were extended.
- The API remains v11. The current ledger deliberately leaves the affected
  Matmul units pending or `needs_recheck` until apply validation writes them back.

## Round 23 — independent rotating senders and Matmul remediation (2026-08-07)

- **Trigger (API-009/Matmul review):** block-sharded Matmul can have shard senders outside its
  output-work receiver rectangle. Widening that rectangle changed traffic, receiver roles, and ACK
  fan-out; the degenerate helper path had also changed the original local-copy primitive.
- **API:** `Mcast1D` and `Mcast2D` now accept ordered rotating sender sets independently of their
  fixed receiver rectangles. Semaphore ownership covers the receiver/sender union, and the existing
  dense-fan-out sentinel lets each device sender derive area-1 ACKs inside the rect or area ACKs
  outside it. The existing v11 CT/RT wire already represented this, so the version remains **11**.
- **Kernel semantics:** degenerate self-only copies again use same-core `noc.async_write` with no
  immediate barrier. The block-sharded reader delegates that case to `SenderPipe::send()` instead of
  duplicating it at the call site.
- **Matmul:** legacy and descriptor factories restore the original receiver rectangles and carry
  shard sender order separately; reuse their descriptor NoCs; use one unconditional `McastArgs` ABI;
  append helper blocks after fixed operation fields; and preserve TensorAccessor chaining. The sparse
  factory received the same shared-kernel ABI treatment after focused triage exposed stale bindings.
- **Correctness:** host build passed; host wire tests 30/30; source audit 17/17; helper normal and
  Watcher suites 80/80 each; full Matmul 816 passed, 310 expected skips, 2 known xfails; sparse Matmul
  18/18. The outside-sender and degenerate helper/production cases passed with the intended kernels.
- **Performance:** at 800 MHz, three warmups and 20 measured records gave +0.643% for 2D SDXL,
  +0.809% for 1D SDXL, and -0.045% for transposed 2D versus matched `4a1d6a97ca9` artifacts. New
  sensitive-path records measured 2,548.925 ns for 1x1 degenerate and 11,787.407 ns for the
  sender-span-greater-than-receiver-span case; both asserted the intended kernel source.

---

## Round 22 — typed Flag control values, API v11 (2026-08-06)

- **Trigger (API-007):** Matmul's sparsity batch-validity exchange carries `VALID` or
  `IGNORE_BATCH` in the data-ready semaphore cell; the v10 helper hardcoded the Flag value to
  `VALID` on both faces.
- **API:** `SenderPipe::send_signal(uint32_t value = VALID)` now writes a caller-supplied non-zero
  Flag value before the existing multicast. Flag `ReceiverPipe::receive_signal()` waits for
  `>= VALID`, captures and returns the observed value, then clears the cell to `INVALID` once.
  Counter remains monotone `+1` and requires the default argument. Handshake behavior is unchanged.
- **Version:** caller-visible control semantics advance `MCAST_PIPE_API_VERSION` **10 → 11**. The
  host wire does not change; the existing v10 fleet becomes stale and re-enters apply-dm-helper as
  Tier 0 before new Matmul migration work.
- **Style:** no new style fork or bake-off cell. The value is protocol payload on the already selected
  Flag path; no performance re-measure was warranted.
- **Validation:** fresh-JIT focused default-`VALID` and `IGNORE_BATCH` cells passed under `--dev`.
  The complete helper suite passed 79/79 and `McastHostFixture.*` passed 28/28.

---

## Round 21 — final v10 release gate (2026-08-05)

- **Validation:** the host build, helper host/device suites, opaque-boundary
  audit, and all mapped Matmul, Conv, GroupNorm, and Sort correctness
  inventories passed. Fresh artifacts cover all 13 migrated kernels and the
  build covers all 12 migrated host bindings.
- **Performance:** nine cases passed directly. Legacy GroupNorm initially
  appeared +2.508% against an older artifact, but a controlled isolated
  checkout reproduced the actual pre-migration baseline `4a1d6a97ca9` at
  `49,694.26516945126 ns` and current at `49,836.38882787317 ns` under the
  same firmware, build, and profiler environment. Current is +0.285996%,
  within the 1.5% gate. The previously passing migrated snapshot measured
  `49,850.05759004791 ns`, confirming no migration-commit regression.
- **Result:** all seven gates are green. The helper remains API v10; API-002
  compile-time sender/receiver-face enforcement remains deliberately deferred.
  RT compaction is no longer part of that open feedback.

---

## Round 20 — GroupNorm production geometry classification (2026-08-05)

- **Trigger (MIG-004):** the fixed three-block GroupNorm sender wire always executes middle, first,
  and last Pipes, so absent edge rectangles add degenerate calls and wrapped groups needed explicit
  performance classification.
- **Classification:** the sharded-v2 factory requires one dense rectangular shard grid. Its mapped
  block- and height-sharded production configurations generate rectangular groups only, so the
  supported production class is zero-edge. No mapped production case reaches a one- or two-edge
  wrapped partition.
- **Coverage:** direct host tests exercise the production splitter with zero-, one-, and two-edge
  coordinate sequences. `GroupNormMcastGeometry` passed 3/3, `McastHostFixture` passed 25/25, and
  `./build_metal.sh` passed.
- **Performance:** the supported zero-edge class reuses the matched SDXL measurements: legacy
  +0.248% and Welford -0.485% versus baseline. Both pass the 1.5% gate; no helper or kernel change was
  needed. API remains v10.

---

## Round 19 — signal-only handshake policy and Sort channel split (2026-08-05)

- **Trigger (API-003/MIG-002):** `send_signal()` and `receive_signal()` ignored the channel's
  handshake bit, forcing Sort to retain a raw reader-ready semaphore around an otherwise migrated
  row-start event.
- **API:** handshaked signal-only sends now wait/reset consumer readiness and receivers acknowledge
  the current sender before waiting. `handshake=false` retains the existing Counter/Flag behavior.
  No call-site spelling changed and no prior handshaked signal-only caller depended on the exception,
  so the rollout remains `MCAST_PIPE_API_VERSION 10`.
- **Sort:** one handshaked row-start Counter channel owns semaphore IDs 0/1; one no-handshake
  sub-stage Counter channel owns ID 2. The raw reader-ready semaphore is removed. The independent
  writer-done counter moves to ID 3 and remains operation-owned.
- **Coverage:** the control-only matrix now covers both handshake policies over 1x2/1x8 and 2/32
  rounds; the complete cold-cache helper suite passed 77/77. The exact cold-JIT long Sort case,
  both `Ht=2` deadlock regressions, and all seven long cases passed.
- **Performance:** three 3-warmup/20-record run medians were 145,201,100.414, 144,983,524.867, and
  145,768,174.937 ns. Their median is 145,201,100.414 ns, +1.195124% versus the
  143,486,262.222 ns baseline and within the 1.5% gate.

---

## Round 18 — self-describing rotating wire, API v10 (2026-08-05)

- **Trigger (API-001):** rotating kernels repeated the host-generated sender span as a third
  `McastArgs` template argument, allowing the CT/RT wire shape and decoder type to disagree.
- **API:** `MCAST_PIPE_API_VERSION` is **10**. The uniform CT block is now
  `[active, data_ready, consumer_ready, num_active, flags, rotating_span]`.
  `McastArgs<CT_BASE, RT_BASE>` derives fixed/rotating mode, receiver type, runtime width, and both
  next offsets from the constexpr sixth word. Zero denotes the fixed four-word RT layout; a nonzero
  value denotes `4 + 2 * rotating_span` RT words.
- **Callers:** all migrated emitters now provide the sixth word and every production kernel uses the
  two-argument decoder. API-002 sender/receiver-face enforcement remains explicitly deferred; RT
  compaction is not tracked as follow-up work.
- **Safety:** Gate 2's opaque-boundary audit was completed before this wire change. A new durable
  audit rejects a third `McastArgs` template argument in any migrated kernel.
- **Validation:** `./build_metal.sh`, `McastHostFixture` 25/25, and the complete helper device suite
  73/73 passed. Fresh-JIT focused cases passed for Matmul, Conv height/block/width, GroupNorm
  legacy/Welford, and Sort. Complete mapped inventories passed at their recorded counts:
  Matmul 302/188; each Conv feature route 48/16 plus three DRAM-config cases and shared DRAM 14/14;
  GroupNorm legacy 108/2, Welford 108/2, fixed/default 19/6; Sort long 7/7 and deadlock 2/2.

---

## Round 17 — non-owning receiver-coordinate view (2026-08-04)

- **Trigger:** SegFormer width-sharded Conv profiling isolated substantial receiver overhead in
  `McastArgs::receiver()`: the rotating `SPAN=18` path first materialized a 36-word local coordinate
  array and then `ReceiverPipe` copied all 36 words into a second owned array.
- **API (API-006):** `ReceiverPipe` now retains a non-owning pointer to its sender-coordinate pairs.
  The pointed storage must outlive the pipe. `McastArgs::receiver()` satisfies that contract by
  pointing directly at the kernel's stable RT-argument block; the raw-construction test keeps its
  local array alive for every pipe use. The durable contract and validation evidence are recorded in
  `api_feedback.md`.
- **Version:** remains `MCAST_PIPE_API_VERSION 9`. Existing array arguments decay to the accepted
  pointer and require no call-site rewrite; this is an internal representation and lifetime-contract
  change.
- **Correctness:** the complete `--dev` helper suite passed 73/73, including rotating spans 2/4/8 and
  the by-hand `ReceiverPipe` construction. The exact SegFormer 576-channel width-sharded nightly node
  passed at PCC 0.9998909 against 0.985.
- **Performance:** three independent real-time-profiler runs measured 38,362.905, 38,377.304, and
  38,414.444 ns. Their median, 38,377.304 ns, is +0.958% versus the immediate pre-migration parent
  (38,013.031 ns), below the 1% investigation threshold and improved from the migrated 38,682.593 ns
  (+1.761%).

---

## Round 16 — caller-managed source-L1 lifetime (2026-08-04)

- **Trigger:** real-time profiling of the SDXL VAE Conv migration attributed most of its regression
  to the remote-only `async_writes_flushed()` performed after every weight and bias multicast. Review
  confirmed that this wait protects payload-source reuse; linked data→signal ordering does not itself
  require the sender to wait after every issue.
- **API (API-005):** added the method-template policy
  `send<SourceL1Guard::CallerManaged>(src_l1, dst_l1, size)`. The existing `send(...)` spelling remains
  source-compatible and defaults to `SourceL1Guard::Guard`, preserving the guarantee that `src_l1` is
  reusable on return. Caller-managed sends require the source to remain unchanged until a later NoC
  completion point. The durable contract and validation evidence are recorded in `api_feedback.md`.
- **Safety:** caller-managed mode skips only the remote-only SENT source-lifetime fence. A real sender
  loopback still waits for ACKed completion, rotating Flag mode still flushes before resetting its
  signal-source cell, and Counter mode still drains multicast atomic acknowledgements.
- **Version:** remains `MCAST_PIPE_API_VERSION 9`. The addition is opt-in and does not stale or rewrite
  any existing caller; the rollout version remains the compatibility key for mandatory migrations.
- **Conv adoption:** the height-sharded weights sender uses caller-managed sends for its weight and
  bias sources. Fully buffered sources need no completion wait; streaming weights flush immediately
  before the next block overwrites the source slot. The full helper send path is `FORCE_INLINE`.
- **Coverage/performance:** added a remote Flag test that reuses one immutable source across four
  caller-managed sends; the complete `--dev` helper suite passed 73/73. The exact SDXL VAE correctness
  node passed at PCC 0.9999325, and its 20-record real-time median improved from 28,719.126 ns to
  28,161.499 ns (+0.736% versus the reverse pre-migration baseline).

---

## Round 15 — width-sharded Conv activation migration (2026-08-03)

- **Trigger:** the earlier v9 port's 25 numerical failures predated the Round-13 ACKed completion
  rule for a real INCLUDE-source loopback, so the production reader was re-entered for a fresh
  end-to-end port rather than left design-blocked.
- **Historical API at this round:** `MCAST_PIPE_API_VERSION 9`. One rotating, handshaked Flag `Mcast2D` owns the full
  reader rectangle while carrying `max(input_cores,output_cores)-1` as the distinct active ACK
  population. The then-current `McastArgs<12,3,num_input_cores>` consumed the actual sender-coordinate
  prefix; Round 18 supersedes this with the self-describing `McastArgs<12,3>` v10 wire.
- **Rollout:** the activation reader and width-sharded factory migrated atomically in `fe866a1d0c4`.
  The raw multicast/semaphore protocol and physical-coordinate lookup arrays were replaced by
  `SenderPipe::send()` and `ReceiverPipe::receive(round)`, removing 102 net production lines.
- **Validation:** host build passed; exact fresh-cache `--dev` route passed at PCC 0.999956503 with
  JIT evidence; the full feature inventory passed 48 cases with 16 legitimate skips; the mapped
  DRAM-config route passed at PCC 0.998234911; post-integration helper coverage passed 72/72.
- **Reconcile:** all 91 census/ledger paths exist; no raw primitive callsite was introduced; totals
  are 13 migrated kernels / 12 migrated host bindings / 78 deferred kernels, with nothing pending,
  quarantined, or marked `needs_recheck`.

---

## Round 14 — sort control-channel migration (2026-08-03)

- **Trigger:** upstream split sort's single return semaphore into independent reader-ready and
  writer-done counters, invalidating the historical design-gap classification.
- **API:** remains `MCAST_PIPE_API_VERSION 9`. The recurring coordinator→workers phase event maps to
  existing no-handshake Counter `send_signal()` / `receive_signal()`; the two return counters remain
  explicit operation protocol.
- **Coverage:** added four control-only Counter cases spanning 1×2/1×8 rectangles and 2/32
  back-to-back signals. Complete helper suite passed 72/72 before and after integration.
- **Rollout:** coordinator and reader migrated through a host `Mcast2D` wire in `7337302b564`.
  Writer received coupled dead-runtime-argument cleanup but remains helper-neutral in the ledger.
- **Validation:** host build passed; exact fresh-cache `--dev` long-tensor case passed with all three
  JIT artifacts; Ht=2 deadlock pair passed 2/2; full long-tensor inventory passed 7/7.
- **Reconcile:** all 91 ledger paths exist, no raw primitive callsite was introduced, and totals are
  12 migrated kernels / 11 migrated host bindings / 79 deferred kernels.

---

## Round 13 — ACKed sender loopback and post-allgather rollback (2026-07-30)

- **Trigger:** review found that the flag receiver's wait proves arrival only
  on remote receivers. A sender that publishes and consumes its own
  INCLUDE-source destination has no corresponding `receive()` wait.
- **API:** remains `MCAST_PIPE_API_VERSION 9`; this is an internal completion
  strengthening. `SenderPipe::fence_(loopback)` waits for ACKed write
  completion when a real loopback copy is emitted and retains the cheaper SENT
  fence for remote-only traffic. Counter signaling still drains its multicast
  atomic acknowledgements.
- **Coverage:** F3 now publishes the looped-back destination immediately to a
  same-core compute kernel for 32 iterations. The complete helper suite passed
  68/68.
- **LayerNorm:** restored the post-allgather sender and receiver together to
  `llk_helper_library`. The accepted non-mcast-1D host path keeps the sender
  outside the receiver rectangle while relying on `MCAST_INCL_SRC`; raw
  `num_blocks` can also differ from `McastRect::area()` for ragged grids.
  Expanding the rectangle or rejecting accepted configurations would be an
  operation-contract change, not a kernel migration.
- **Rollback validation:** host build passed; one exact post-allgather smoke
  passed; all four mapped nodes passed 136/136 with the restored pair. Those
  cases exercise only `mcast_1d`, so they validate the rollback but do not
  unblock the migration.
- **Commit:** `307951cc8dc`.

---

## Round 12 — v9 remediation of split-count and multi-rectangle senders (2026-07-29)

- **Trigger:** review follow-up for Blackhole source-L1 lifetime, the Conv 1D
  split acknowledgement count, dead Conv ABI words, and the two GroupNorm v2
  senders.
- **API:** remains `MCAST_PIPE_API_VERSION 9`. The GroupNorm acknowledgement
  gate remains raw because it protects remote L1 gathering, while the helper
  owns only the no-handshake payload/ready multicasts.
- **Conv:** migrated the 1D sender using a runtime active-ack count; made 2D
  geometry-derived dense acknowledgement explicit; removed dead count and
  runtime-semaphore words from both host/device ABIs.
- **GroupNorm:** migrated legacy and Welford v2 senders with raw pre-gather
  acknowledgement wait/reset plus no-handshake middle/optional rectangle
  pipes; mapped inventories validate the host-generated receiver partition.
- **Validation:** helper 68/68; HEIGHT and BLOCK Conv inventories 49 passed,
  16 expected skips each; DRAM Conv 14/14; legacy and Welford GN inventories
  108 passed, 2 expected skips each; fixed/default GN nodes 19 passed, 6
  expected skips; host build passed. Fresh JIT paths were recorded for every
  migrated pair.

---

## Round 11 — v9 semantic port onto llk_helper_library (2026-07-29)

- **Trigger:** port the 19 July production migrations without rebasing over
  substantial intervening TT-Metal changes.
- **Baseline:** `origin/llk_helper_library` at `54d8dfb7bef`.
- **API:** current `MCAST_PIPE_API_VERSION 9`; helper implementation retained.
- **Result:** 9 migrations completed and fully validated; 10 source migrations
  rejected or reverted because v9 cannot own their complete protocols.
- **Validation:** helper 68 passed; post-allgather LayerNorm 136 passed;
  GroupNorm 240 passes across legacy/Welford plus expected skips; matmul in1
  302 passed with 188 expected skips; fixed Conv families 98 passes plus 32
  expected skips and 14 DRAM regressions; host rebuild passed.
- **Newly explicit gaps:** typed control values, acknowledged signal-only,
  one-gate/multi-block mixed-mode streaming, race-free no-handshake receiver
  initialization, independent data/signal loopback, and sender-side loopback
  destination completion.
- **Tracking:** `migration/ledger.json`, `ledger.md`, `test_map.json`,
  the archived rollout report, `archive/reconciliation/reconcile_2026-07-29.md`,
  and per-kernel migration logs.

---

## Round 1 — tune-helper + apply-helper (2026-06-04 / 06-05)

- **Trigger:** standalone tune-helper run for the recurring NoC-multicast + semaphore
  handshake block; then apply-helper rollout.
- **API delivered:** `Pipe<MCAST, STAGING, PRE_HANDSHAKE, LINK>` + `McastRect{x0,y0,x1,y1,num_dests}`.
  Caller picks `MCAST` (EXCLUDE_SRC/INCLUDE_SRC) and a matching `num_dests`.
- **Bake-off winners baked in:** flush fence (−27%), level flag (−29%), linked pair (−36%),
  INCLUDE_SRC loopback for sender-in-rect (+26–41%).
- **Rollout:** 13 kernels migrated (atomic commits, mapped tests green), 7 reverted under
  `--mode=run-all`. 310 lines of open-coded mcast removed.
- **Artifacts:** `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp`,
  `tests/ttnn/unit_tests/kernel_lib/test_mcast_pipe.py` (45/45),
  `helper_design/mcast_pipe/*`, `migration/*`.
- **Known limits flagged:** single-rect `num_dests` couldn't express
  data-mcast-population ≠ handshake-ACK-count (conv-1D weights sender HUNG); senders
  migrated worse than receivers; R6 role-flip / R4 streaming / CCL deferred.

---

## Round 2 — drop `MCAST`, add active-core count (in progress)

- **Trigger:** review feedback — *"don't expose mcast mode; you're missing the number of
  active cores as a parameter — infer the modes from that."*
- **Decisions:**
  - D1: **two counts** — `McastRect` = pure geometry (area = data-mcast destination
    population); new ctor arg `num_active_cores` = handshake ACK count. Fixes the round-1
    population≠ACK divergence (conv-1D).
  - D2: **runtime mode inference** — Pipe compares its own NoC coords to the rect; deletes
    the `MCAST` template param and the `num_dests` field.
- **API after:** `Pipe<STAGING, PRE_HANDSHAKE, LINK>` + `McastRect{x0,y0,x1,y1}` +
  `Pipe(noc, dest, num_active_cores, data_ready, consumed)`.
- **Artifacts touched:** helper, unit test/bake-off kernels, 13 call sites, design docs,
  `migration/` report, memory.
- **Verification — Phase 2 unit gate (Blackhole p150a):** `test_mcast_pipe.py` **45/45 PASS**.
  The mode knob is gone, so the same suite now doubles as the runtime-inference gate and
  proves IR1 (coord-space comparison): EXCLUDE inferred (sender out-of-rect: coverage/smoke/
  pre_handshake), INCLUDE_SRC loopback inferred (sender in-rect: f3_loopback), degenerate
  self-only collapsed to local copy (num_active_cores==1: f3_degenerate). No hangs.
- **Status:** helper + unit gate done (committed). 13 call sites (Phase 3) pending review.

### Round 2 — inference refinement (Phase 3 finding)

Phase 3 migrating matmul exposed a flaw in D2 as first implemented. The first rule was
`loopback iff sender_in_box`, with `num_dsts = rect.area()`. That hung matmul 1d: the in0
sender sits at the **top-left corner of its own broadcast box** but uses EXCLUDE_SRC — it
already holds the in0 block as its mcast source and must not self-overwrite. "Sender in box"
alone cannot tell EXCLUDE-in-box (matmul) from INCLUDE (conv-WS round-robin).

**Refined rule (still no mode knob, still inferred):**
- `num_active_cores` is the **recipient count** = the NoC `num_dsts` for data+flag = the ACK
  count. (Confirmed against the proven raw matmul: it mcast to `in0_mcast_num_cores` with the
  comment *"num_dests must not include source"*; round-1 used that same value for the wait and
  passed → recipients == acks in the tested configs.)
- **loopback (INCLUDE_SRC) iff `sender_in_box && num_active_cores == rect.area()`** — the sender
  is in the box AND counted as a recipient. matmul: 15 recipients ≠ 16-core box → EXCLUDE ✓;
  conv-WS: readers == box → INCLUDE ✓.
- `rect.area()` is used ONLY for that test, never as a transfer count.

Gap the unit suite missed (sender-in-box + EXCLUDE) is now covered by the matmul mapped test;
a synthetic unit case is a follow-up.

### Round 2 — Phase 3 migration progress (per family, mapped-test gated, --mode=halt)
- **matmul (4 kernels):** in0 sender/receiver, in1 sender/receiver. 1d + 2d PASS. ✓
- **conv (3 migrated + 1 reverted):** 1d-receiver (HS) PASS; 2d sender+receiver (BS) PASS;
  **width-sharded activation sender REVERTED to raw** — un-inferable partial-box self-gather,
  raw test PASS. ✓
- **groupnorm (2 kernels):** reduce-receiver (legacy) + welford-receiver PASS. ✓
- **topk (1 kernel):** reader_final_topk send_signal PASS. ✓
- **layernorm (1 kernel):** reader_mcast_sender_unary_sharded_ln send_signal PASS. ✓

**Phase 3 result: 11 kernels migrated to the new API (all mapped tests green), 1 (conv-WS)
intentionally kept raw + documented.** All on BH p150a, single parametrization each.

## Round 3 (2026-06-10) — loopback from src/dst aliasing; area() retired

- **Trigger (user):** the `num_active == area()` membership proxy is "crooked" (coordinate-space
  contract + integer aliasing). Discussion ground truth: the R6 block-sharded kernel proves the
  mode is per-core and resolves as membership ∧ buffer-aliasing — and an in-box-but-inactive
  sender takes INCLUDE harmlessly (self-write = dead store into landing memory nobody reads).
- **New rule (per send, no knob):** `loopback iff sender_in_rect_() && src_l1 != dst_l1`.
  `src == dst` means the sender's copy is already in place (matmul in0, R6 extract path) — an
  overlapping self-loopback is unspecified. The flag mcast rides the same mode (INV4);
  `send_signal()` (no data) stays EXCLUDE.
- **Count convention:** `num_active_cores` NEVER counts the sender; loopback paths add +1
  internally (API counts self there). Degenerate self-only guard is `num_active == 0 && in box`.
- **`McastRect::area()` deleted** — the rect is pure routing geometry again.
- **conv-WS becomes expressible** (recipient, src != dst → INCLUDE): migration is the natural
  next step; not done this round.
- **Out of inference reach:** loopback FLAG with src == dst data (R6 role-flip extract arm —
  flag INCLUDE, data EXCLUDE). Stays raw.
- **Harness fix:** test_mcast_pipe.py used hardcoded VIRT_X/Y=(1,2); machine firmware now maps
  worker (0,0)→virtual(18,18) (translated coords) — every test mcast targeted empty coords →
  hang even at smoke. Now uses `device.worker_core_from_logical_core` (the binding DOES exist).
- **Verification (BH p150a):** unit 45/45 PASS; mapped tests of all 11 migrated kernels green
  (matmul 1d+2d, topk, conv HS+BS conv_features 48 each, gn legacy+welford, sharded-LN ×32,
  deepseek sampling). conv-WS raw untouched.

### Round 3 — conv-WS migrated (12th kernel)

- `activation_reader_width_sharded`: round-robin self-gather data+flag broadcast → one
  `Pipe::send()` (re-applied round-1 diff under the inferred-mode API; PRE_HANDSHAKE=false;
  readiness counter + receiver-branch ack stay raw, they count num_mcast_cores not readers).
- **Bug found via WS mapped test (PCC 0.92):** WS factory swaps the rect start/end for NOC1
  (`conv2d_op_width_sharded_program_factory.cpp:355`); `sender_in_rect_()` assumed x0<=x1 →
  EXCLUDE inferred → sender skipped its own copy. Fix: normalize bounds in the membership test
  (the mcast address keeps NoC ordering). Loopback inference was unreachable before the
  src!=dst rule, so round-2 kernels could never hit swapped rects with INCLUDE.
- Verified: conv WS 48/48 (PCC 0.9975 == raw); unit 45/45; matmul 1d + 2d green after the
  sender_in_rect_ change.

### Round 3 — R6 role-flip migrated: matmul block-sharded in0 sender_receiver (13th kernel)

- Two faces on every grid core: ONE sender Pipe (PRE_HANDSHAKE, num_active = num_dests-1
  in-grid / num_dests out-of-grid; factory guarantees num_dests==num_cores) + a per-round
  receiver Pipe (`single_core(remote_sender[block_id])` — the ack target rotates).
- The mode table that blocked Round 2 falls out of the src!=dst rule: extract (src==dst) ->
  EXCLUDE n-1; non-extract -> INCLUDE n; out-of-grid -> EXCLUDE n; the raw flag-INCLUDE arm is
  matched by send()'s local VALID set; sender no longer waits its own flag (always-true wait
  dropped). Top-of-loop INVALID reset stays raw — clears the stale VALID from own sender round.
- In-grid single-core collapses to Pipe's degenerate local copy (no handshake/flush), same as raw.
- ~110 lines of open-coded mcast removed. Verified: in0_in1_bias_sharded + sharded_matmul
  suites, 270 tests green, 29 JIT config variants of the kernel dispatched fresh. Untested:
  fused-op (multi-device CCL).

## Round 4 (2026-06-13) — API review: split objects, full recipient count, sem ids+init, noc 2.0

- **Trigger (user):** four API-review bullets (`feedback.txt`). Implementation + both skills
  (`tune-helper`, `apply-helper`) + docs fixed; **migrations deferred** pending review.
- **API before:** one `Pipe<STAGING, PRE_HANDSHAKE, LINK>` with `send/receive/send_signal/
  receive_signal`; ctor `Pipe(noc, McastRect, num_active_cores, Semaphore<> data_ready,
  Semaphore<> consumed)`; receiver constructed `McastRect::single_core(sender)` + `num_active=1`.
- **API after:** two types —
  - `SenderPipe<STAGING, PRE_HANDSHAKE, LINK>(noc, McastRect dest, uint32_t num_active_receiver_cores,
    uint32_t data_ready_sem_id, uint32_t consumed_sem_id)` with `send()` / `send_signal()`;
  - `ReceiverPipe<STAGING, PRE_HANDSHAKE>(noc, uint32_t data_ready_sem_id, uint32_t consumed_sem_id)`
    with `receive(sender_x, sender_y)` / `receive_signal()`.

- **P2 — split sender/receiver into two objects.** A receiver never multicasts, so it carried a dead
  rect + `num_active=1`. `ReceiverPipe` drops both; the sender coords it needs for its R->S ack are
  now a `receive()` argument. `McastRect::single_core` deleted (only the receiver used it).
  *Skills:* tune-helper Step ★ ("asymmetric faces want separate types") + Step F; apply-helper Phase 1
  materialization invariant #2.
- **P3 — `num_active_cores` → `num_active_receiver_cores`, now the FULL count incl. sender-if-receiver.**
  Round 3's convention ("never counts the sender; loopback adds +1 internally") forced the caller to
  pre-subtract. New: the caller states the whole recipient set; the SenderPipe derives
  `ack_count = N - (sender_in_rect?1:0)` and `mcast_dests = loopback ? N : ack_count`. Degenerate guard
  is now `ack_count == 0`. Net mapping: old `num_active_cores` == new `ack_count`; old `+1` loopback ==
  new `N`. *Skills:* tune-helper Step ★ ("count statable from caller topology alone"); apply-helper
  Phase 1 invariant #3.
- **P4 — ctors take semaphore IDs and own init.** Was: caller passed pre-built `Semaphore<>` and (e.g.
  toy/matmul) pre-set VALID. Now: ctors take `uint32_t` ids, construct `Semaphore<>` internally, and
  init the cell THIS side waits on (SenderPipe: `consumed = 0` under PRE_HANDSHAKE; ReceiverPipe:
  `data_ready = INVALID` under Staging::Flag). The other side's cell is left to that side's ctor — no
  cross-core init race. Host `CreateSemaphore` still allocates the ids. *Skills:* tune-helper Step ★/F;
  apply-helper Phase 1 invariant #4.
  - **Follow-up (user review):** the SenderPipe ctor also folds in the sender's local data-ready
    pre-set — a 6th `initial_ready` ctor arg, **default `VALID`** (the dominant pattern: 5/6 migrated
    data senders did `<flag_sem>.set(VALID)` before the loop). A signal sender that starts INVALID
    (sharded-LN phase-1) passes `initial_ready = INVALID`. Migrating call sites drop their manual
    pre-loop `set(VALID)` line. No-op for Staging::Counter.
- **P5 (user review) — drop the `LINK` template param; always link.** Census of all 16 `Pipe<>`
  instantiations: **none** override `LINK` (the two non-default ones set only `PRE_HANDSHAKE=false`).
  Per the helper's own "single-path is the default; a dual-path must earn its place" rule, the
  `LINK=false` (unlinked + barrier-between) arm is removed and the data mcast is always issued
  `linked=true` (flag terminates the chain with `linked=false`). `SenderPipe<STAGING, PRE_HANDSHAKE>`
  is now 2 template params. *Skills:* tune-helper Step E.4 + Step F; apply-helper Phase 1 invariant #5.
  - **The supposed unlinked consumer (sdpa read_k) doesn't actually need unlinked — corrected finding.**
    The `LINK=false` arm was justified in `proposed_helpers.md`/`design/style_bakeoff.md` by *"a barrier is
    structurally required between data and flag, e.g. sdpa read_k."* On inspection that is **wrong**:
    - The kernel is `sdpa_decode/device/kernels/dataflow/dataflow_common.hpp::read_k`, the `do_mcast`
      sender branch (~L631–653). It does `noc_async_write_multicast(..., /*linked=*/false)` → **full
      `noc_async_write_barrier()`** → `noc_semaphore_set_multicast(...)` → barrier. That is the slow,
      conservative pattern: it waits for the data to fully ACK before signaling.
    - There is no structural obstacle to linking. Data + flag target the **same vertical column**
      (`get_noc_multicast_addr(mcast_x, y0, mcast_x, y1, ...)`) on the same NoC/VC; same-VC FIFO order
      (INV4) already gives the receiver data-before-flag, so the full barrier is overkill.
    - **sdpa's OWN `chain_link.hpp` proves it** — for the same K/V chunk broadcast it does exactly the
      helper's pattern: `noc_async_write_multicast(..., /*linked=*/true)` → `noc_semaphore_set_multicast`
      → `noc_async_writes_flushed()` (chain_link.hpp L231–233). And the matmul/conv senders link the
      identical data-write→sem-set sequence.
    - So `read_k` is a **`refactor`** (census tags it exactly that), not a `defer`, and it would *gain*
      the −36% linked win — it never needed `LINK=false`. The arm had **no genuine consumer at all**,
      which is the strongest possible reason to delete the knob. (Caveat: this is a code-reading
      conclusion; the rigorous confirmation is to migrate read_k to the linked helper and run the
      sdpa_decode suite. If some *future* kernel must genuinely fence between data and flag, re-add the
      unlinked arm as a refinement then.)
- **P6 (user review) — push compile-time, core-uniform ctor args to TEMPLATE params.** Audited each
  `uint32_t` against the kernels (matmul `reader_bmm_tile_layout_in0_sender_padding.cpp` is the witness):
  - **Templatable (compile-time + identical across all cores running the binary):** `num_active`
    (`get_compile_time_arg_val(17)`), the two sem ids (`get_compile_time_arg_val(15/16)`), and
    `initial_ready` (a literal). P3 also made `num_active` core-uniform (the in-grid/out-of-grid ±1 is
    now derived internally), so it's safe to bake in. → moved to template params.
  - **Hard-runtime (must stay):** `McastRect` coords — `get_arg_val` in matmul because each row-sender
    in a 2D grid targets a *different* rect under one compiled binary (per-core variation); plus
    they're device-resolved virtual coords. `send(src,dst,size)` — CB pointers, per-iteration.
    `receive(sender_x,sender_y)` — varies per receiver (2D) and rotates per block (R6). → stay runtime.
  - **API after:**
    `SenderPipe<NUM_ACTIVE_RECEIVER_CORES, DATA_READY_SEM_ID, CONSUMED_SEM_ID, STAGING=Flag,
     PRE_HANDSHAKE=true, INITIAL_READY=VALID>(noc, dest)` and
    `ReceiverPipe<DATA_READY_SEM_ID, CONSUMED_SEM_ID, STAGING=Flag, PRE_HANDSHAKE=true>(noc)`.
    The ReceiverPipe ctor now takes only `noc`. Perf upside is marginal (`get_semaphore(ID)` folds to a
    constant address, but `ack_count` stays runtime via the `sender_in_rect_()` membership check); the
    real win is type honesty — the host cannot pass a per-core-varying value where a uniform one is
    required. *Skills:* tune-helper Step ★ (arg-classification rule); apply-helper Phase 1 invariant #6.
  - **Migration impact:** every call site moves these values from ctor args to template args — the
    sem ids/count/initial_ready (already `get_compile_time_arg_val` in the kernels) become template
    args; only the rect stays a ctor arg.

### Round 4 — Tier 2 matmul + P3 RESET to recipient-count semantics (user review)

- **P3 conflict found at matmul:** the shared `reader_bmm_tile_layout_in0_sender_padding.cpp` is used
  by the 1D factory (sender IN rect) and the 2D factory (sender OUT of rect). P3-as-implemented had
  the helper *subtract* 1 (runtime `sender_in_rect`) and the caller pass the *rect-population* count —
  but that compile-time count differs per topology (1D needs `num_dests+1`, 2D needs `num_dests`) and a
  shared kernel has no compile-time discriminator. Bisection confirmed no single constexpr works.
- **Decision (user): RESET P3 to recipient-count semantics** (the round-3 direction):
  `NUM_ACTIVE_RECEIVER_CORES` = the RECIPIENT count = EXCLUDE_SRC `num_dests` = ACK count — the value
  every factory ALREADY computes. The helper no longer subtracts; it ADDS +1 only for INCLUDE
  loopback. The in0 sender now passes `in0_mcast_num_dests` verbatim and is correct for BOTH 1D and 2D
  with **zero host-factory edits**. (Softens P3's "caller passes full count incl. sender" wording, but
  keeps "helper decides num_dests per mode".) Helper §send/send_signal + docstring updated; tune/apply
  skills' P3 lesson should be read with this correction.
- **Migrated matmul (5):** in0 sender, in1 sender(+bias), in0 receiver, in1 receiver, R6 role-flip
  block-sharded (persistent SenderPipe + per-round ReceiverPipe, rotating `receive(sx,sy)`).
  Verified BH p150a: unit 39/39, toy 4/4, matmul 1D mapped + 2D multiple-output 56/56 (incl. R6).
- **Migration rule for the remaining families (conv/gn/topk/ln):** pass the kernel's existing
  recipient count (the old `num_active` ctor value = factory `num_dests`) **verbatim** as the template
  `NUM_ACTIVE_RECEIVER_CORES` — no ±1. Sem ids → template; `receive(sx,sy)`; drop the manual pre-loop
  `set(VALID)` (ctor owns it via INITIAL_READY; a signal sender that pre-set INVALID uses
  `INITIAL_READY=INVALID`).

### Round 4 — Tier 0 migration (unit test) + P4 correctness fix

- Migrated the 3 unit-test kernels (`pipe_sender/receiver/f3_sender.cpp`) + `test_mcast_pipe.py` to the
  `SenderPipe`/`ReceiverPipe` template API; dropped the `flag_unlinked`/`LINKED` axis (LINK gone);
  F3 count `R-1` → `R` (P3: now includes the sender).
- **P4 BUG found by the pre_handshake hang (4/39 failing) and fixed:** the SenderPipe ctor's
  `consumed_.set(0)` **raced** with receivers' `consumed_.up()` — a receiver acks before the sender's
  ctor runs, the `set(0)` clobbers the ack, the sender's `wait(ack_count)` hangs forever. Root cause:
  a counter that **remote cores increment** has no happens-before with the waiting side's ctor, so it
  CANNOT be kernel-initialized — its initial 0 must come from host `CreateSemaphore(..., 0)` (every
  call site already does this). Fix: removed the `consumed` ctor init; kept only the race-free local
  inits (receiver's own `data_ready`, sender's own broadcast `INITIAL_READY`). Corrected the P4 lesson
  in the header + both skills. **Unit suite 39/39 PASS** (BH p150a).
- **P1 — noc 2.0 only; no raw mcast free functions.** Round 3 still called raw
  `noc_async_write_multicast` / `_loopback_src` + open-coded `::get_noc_multicast_addr`, and a raw
  `noc_async_read`/`get_noc_addr` in the self-copy — despite the docstring claiming "object API." Now
  the data mcast goes through `Noc::async_write_multicast<McastMode>` with `UnicastEndpoint` +
  `MulticastEndpoint`, and the self-copy through `Noc::async_read`. Flag mcast was already on
  `Semaphore<>::set_multicast` / `inc_multicast`. *Skills:* apply-helper Phase 1 invariant #1
  ("object API only; a missing overload is a gap to flag, not a license to drop to raw"); tune-helper
  Step F implementation-contract commitment.

- **Migration impact (DEFERRED — to migrate after review):** every call site that used `Pipe<>` must
  move to `SenderPipe`/`ReceiverPipe`, pass sem ids instead of `Semaphore<>` objects, drop manual sem
  init, and pass the full recipient count. Affected: 13 committed kernels (matmul ×5 incl. R6,
  conv ×4, groupnorm ×2, topk, layernorm), 2 untracked toy_matmul kernels, and the 3 unit-test
  kernels (`pipe_sender.cpp`, `pipe_receiver.cpp`, `pipe_f3_sender.cpp`) + `test_mcast_pipe.py`.
  Until migrated, those kernels reference the removed `Pipe` type and will fail to JIT-compile.
- **Verification:** none yet on device (header-only change, no rebuild; kernels compile at JIT/test
  time during migration). Unit gate + mapped-test re-run is the first migration step.

### Round 4 — Tier 2 COMPLETE (all 13 production kernels migrated + verified, BH p150a)

- **matmul (5):** in0/in1 sender+receiver + R6 block-sharded. unit 39/39, toy 4/4, 1D mapped + 2D 56/56.
- **topk (1):** `reader_final_topk` send_signal. `test_topk` W=8192 PASS.
- **layernorm (1):** `reader_mcast_sender_unary_sharded_ln` send_signal, `INITIAL_READY=INVALID`.
  `test_layer_norm_sharded_single_stage` welford PASS.
- **groupnorm (2):** `reader_mcast_receiver` + `welford_reader_mcast_receiver` (v2 block-sharded) PASS.
- **conv (4):** width-sharded activation sender + the 3 WEIGHTS kernels (1D recv, 2D send/recv). The
  weights kernels read sem ids from RUNTIME args — P6 (template sem ids) didn't hold, so (user
  decision) the conv2d sharded factory was edited to APPEND the 2 weights sem ids + the 2D sender
  recipient count as compile-time args to `writer_compile_time_args` (no CT-index shift; runtime args
  left in place). Required a `build_metal.sh` rebuild. `test_conv_features` HS + BS PASS.
- **Two API-reality findings resolved this tier:** (1) a sender kernel SHARED across topologies with
  differing in-rect-ness (matmul in0, 1D vs 2D) → reset P3 to recipient-count semantics so the same CT
  count works for both with no host edit; (2) RUNTIME-sourced sem ids (conv weights) → host-promote to
  compile-time. P3/P6 in the helper docstring + tune/apply skills carry these caveats.
- **Status: Round 4 migration COMPLETE.** 13 production + 2 toy_matmul + 3 unit kernels on the new
  `SenderPipe`/`ReceiverPipe` API; no old `Pipe<>` / `McastRect::single_core` usage remains.

---

## Reentrancy infrastructure — version stamp + migration ledger (2026-06-19)

- **Trigger:** make `apply-dm-helper` re-entrant so that when `tune-dm-helper` bumps the API in a future
  round, already-migrated kernels can be **remigrated** automatically, then the not-yet-migrated backlog
  resumed — with durable in-repo state of what's done vs owed.
- **Version stamp:** added `#define MCAST_PIPE_API_VERSION 4` to `mcast_pipe.hpp` (= the Round-4
  `SenderPipe`/`ReceiverPipe` API). `tune-dm-helper` Step G.4 now owns bumping it on every *caller-facing*
  change (and leaving it for internal-only changes).
- **Ledger bootstrapped:** `migration/ledger.json` (+ `ledger.md` mirror) — 66 census sites: **13
  migrated@v4** (set derived by grep of `SenderPipe`/`ReceiverPipe` usage, ground truth — not prose),
  **46 pending** (clean/refactor not yet migrated, incl. the conv 1D-weights sender, gn/ln senders,
  sdpa, CCL family), **7 deferred** (`defer`/`oos`/`ref`). Staleness is *derived*
  (`migrated_api_version < CURRENT`), never stored.
- **Skills updated:** `apply-dm-helper` (Gate-0 fresh-vs-re-entry branch, incremental Phase-1 map,
  Tier-0 = remigrate-stale, per-kernel ledger write-back, report regenerated from the ledger);
  `tune-dm-helper` (Step G.4 version stamp + materialization invariant #7 + exit-checkpoint report).
- **Next API bump → next run:** `tune-dm-helper` bumps `MCAST_PIPE_API_VERSION` to 5; re-invoke
  `apply-dm-helper helper_design/mcast_pipe/ --mode=…` → it remigrates the 13 stale kernels first, then
  continues the 46 pending. No manual re-run of the whole fleet.

---

## Round 5 — naming + `McastRect` NoC-id templating (2026-06-19)

- **Trigger:** `tune-dm-helper feedback.txt` — three claims: (1) `McastRect::start_end_for_noc()` runs a
  corner comparison + per-NoC swap on every `send()` (twice/send) though the NoC id is compile-time —
  template the rect on the NoC id and precompute in the ctor; (2) `Staging` is not a clear name; (3)
  `INITIAL_READY` is not a clear name — make the flag-only scope obvious.
- **Re-entry routing (batched, upstream-first):** item 1 → **Step D** (contract: type signature + where a
  value is computed); items 2,3 → **Step F** (wording). Leftmost = D → one forward pass D→E→F→G. **Step E
  was a re-confirm no-op** (item 1 touches no style fork; coverage/perf maps stand) — **no device bake-off.**
- **Decisions:**
  - **D1 — `McastRect<uint8_t NOC_ID = noc_index>`.** Adds a compile-time `NOC_ID` template param
    (default `noc_index`; factory may pass an explicit id). The four coords stay runtime (per-core). The
    **ctor** computes & stores the routing-correct `(start_x,start_y,end_x,end_y)` for `NOC_ID` once; the
    per-call `start_end_for_noc(noc_id)` method is **deleted** (now a stored-field accessor `bounds()`).
    `SenderPipe` gains a matching `NOC_ID` param so `sender_in_rect_`'s `my_x[noc_.get_noc_id()]` folds to
    a compile-time `my_x[NOC_ID]`.
  - **F2 — `Staging` → `HandshakeKind`** (members `Flag`, `Counter` unchanged). *(user pick)*
  - **F3 — `INITIAL_READY` → `INITIAL_FLAG_VALUE`** — the name now states it's a flag value, hence
    `HandshakeKind::Flag`-only. *(user pick)*
- **API before:** `SenderPipe<N, DR, C, Staging=Flag, PRE_HANDSHAKE=true, INITIAL_READY=VALID>(noc, McastRect{...})`,
  `ReceiverPipe<DR, C, Staging=Flag, PRE_HANDSHAKE=true>(noc)`, `McastRect{x0,y0,x1,y1}`.
- **API after (API version 5):** `SenderPipe<N, DR, C, HandshakeKind=Flag, PRE_HANDSHAKE=true,
  INITIAL_FLAG_VALUE=VALID, NOC_ID=noc_index>(noc, McastRect<>{...})`,
  `ReceiverPipe<DR, C, HandshakeKind=Flag, PRE_HANDSHAKE=true>(noc)`, `McastRect<NOC_ID=noc_index>{x0,y0,x1,y1}`.
- **`MCAST_PIPE_API_VERSION` 4 → 5** (caller-facing: renamed enum + renamed param + `McastRect` type now
  templated → every migrated call site is rewritten). All 13 Round-4 migrated kernels are now **stale@v4**.
- **Artifacts touched:** `design/api_feasibility.md` (Round-5 addendum), `design/style_bakeoff.md` (E no-op note),
  `proposed_helpers.md` (header), this changelog, `mcast_pipe.hpp` (materialized), the 3 unit-test kernels
  + `test_mcast_pipe.py` (ported to the new API).
- **Verification:** header-only + JIT kernel change (no `build_metal.sh` rebuild). `test_mcast_pipe.py`
  unit gate is the green re-confirm of the materialization. Provisional dual-paths: none new (F4 linking
  stayed baked-in; no fork re-decided).
- **Hand-off:** re-invoke `apply-dm-helper helper_design/mcast_pipe/` → Tier-0 remigrates the 13
  stale@v4 kernels to v5 first, then resumes the 46 pending. No manual fleet re-run.

---

## Round 6 — flag-set lifecycle, naming, arg order, comment cleanup (2026-06-20)

- **Trigger:** `tune-dm-helper feedback.txt` — six claims: (1) the per-send local `data_ready.set()`
  is needed only off the loopback path, and there only ONCE (not per send); (2) `INITIAL_FLAG_VALUE`
  is dead weight (the per-send `set` always overwrote the ctor init), drop it but keep a ctor
  `set(VALID)` for the no-loopback case; (3) rename the `consumed` semaphore → `consumer_ready`;
  (4) `HandshakeKind` is a bad name (reads like `PRE_HANDSHAKE`) → `DataReadySignal`; (5) reorder the
  SenderPipe template args; (6) rewrite comments for the FINAL API only (drop round/version
  archaeology, deleted-method and obsolete-template-arg references).
- **Re-entry routing (batched, upstream-first):** items 1,2,5 → **Step D** (contract: signature +
  param order + count/flag semantics); items 3,4,6 → **Step F** (wording/rename). Leftmost = D → one
  forward pass D→E→F→G. **Step E was a re-confirm no-op** (none of the six touches a style fork —
  flush/barrier, flag/counter, linked/unlinked — or adds a matrix cell; coverage/perf maps stand) —
  **no device bake-off.**
- **Root cause for items 1+2 (confirmed in code, not asserted):** `Semaphore<>::set_multicast`
  (`noc_semaphore.h:165`) broadcasts the sender's **local cell** as its source — it takes NO `value`
  argument. For the Flag signal that source is always `VALID`, so it is correctly set **once** in the
  ctor and reused every send; the per-send `data_ready.set()` was redundant. This is exactly the
  proven raw matmul pattern: `reader_bmm_tile_layout_in0_sender_padding.cpp:53` sets the local cell
  `= VALID` ONCE before the loop, then mcasts each iteration. `INITIAL_FLAG_VALUE` could therefore
  never reach the wire (the per-send set clobbered it) → dropped. The loopback path needs no local set
  at all (its INCLUDE-source mcast writes the sender's own cell). The lone `INITIAL_FLAG_VALUE=INVALID`
  consumer (sharded-LN phase-1 signal sender) is unaffected: it never reads its own cell as a flag and
  phase-2 explicitly re-sets that cell (`reader_mcast_sender_unary_sharded_ln.cpp:276`), so always
  ctor-setting `VALID` is correct.
- **Decisions:**
  - **D1 (items 1+2) — flag-set lifecycle.** Drop the `INITIAL_FLAG_VALUE` template param. The ctor
    sets the sender's local data-ready cell `= VALID` once (Flag signal only). `send()` no longer does
    a per-send local set — `signal_ready_` just `set_multicast`s the persistent local `VALID`.
  - **D2 (item 5) — SenderPipe template arg order** is now `NOC_ID` (no default, first) → sem ids →
    `NUM_ACTIVE_RECEIVER_CORES` → `DATA_READY_SIGNAL` (default Flag) → `PRE_HANDSHAKE` (default, last).
  - **F1 (item 3) — `CONSUMED_SEM_ID` → `CONSUMER_READY_SEM_ID`**, member `consumed_` →
    `consumer_ready_` (both faces).
  - **F2 (item 4) — `HandshakeKind` → `DataReadySignal`** (members `Flag`, `Counter` unchanged); the
    `HANDSHAKE` param → `DATA_READY_SIGNAL`. Disambiguates from `PRE_HANDSHAKE`.
  - **F3 (item 1, follow-on) — `send_signal` loses its `value` param** (user pick): since
    `set_multicast` broadcasts the local cell and the ctor seeds `VALID`, `send_signal()` is a plain
    doorbell. No in-scope caller passed non-`VALID` (topk + sharded-LN both use `VALID`; the
    value-carrying moe_gpt is deferred and reads its own cell), so the param was a footgun (would
    silently broadcast `VALID`) — dropped per materialization invariant #5.
  - **F4 (item 6) — comments rewritten for the final API**: removed round-number archaeology
    (Round 4/5, R6, F1/F2/F4 codes as narrative), the deleted-`start_end_for_noc` mention, the
    "include/exclude-src template arg" leftovers, and the long sdpa-read_k linking back-story.
- **API before:** `SenderPipe<N, DR, C, HandshakeKind=Flag, PRE_HANDSHAKE=true, INITIAL_FLAG_VALUE=VALID,
  NOC_ID=noc_index>(noc, McastRect<>{...})`, `ReceiverPipe<DR, C, HandshakeKind=Flag, PRE_HANDSHAKE=true>(noc)`.
- **API after (API version 6):**
  `SenderPipe<NOC_ID, DATA_READY_SEM_ID, CONSUMER_READY_SEM_ID, NUM_ACTIVE_RECEIVER_CORES,
   DataReadySignal=Flag, PRE_HANDSHAKE=true>(noc, McastRect<NOC_ID>{...})`,
  `ReceiverPipe<DATA_READY_SEM_ID, CONSUMER_READY_SEM_ID, DataReadySignal=Flag, PRE_HANDSHAKE=true>(noc)`,
  `send_signal()` (no arg). `McastRect<NOC_ID=noc_index>` unchanged.
- **`MCAST_PIPE_API_VERSION` 5 → 6** (caller-facing: removed param + renamed enum/param + reordered
  template args + `send_signal` signature → every migrated call site is rewritten). All Round-4/5
  migrated kernels are now **stale@v5**.
- **Artifacts touched:** `design/api_feasibility.md` (Round-6 addendum), `design/style_bakeoff.md` (E no-op note),
  `proposed_helpers.md` (header), this changelog, `mcast_pipe.hpp` (materialized), the 3 unit-test
  kernels (`pipe_sender`/`pipe_receiver`/`pipe_f3_sender`) + `test_mcast_pipe.py` (ported).
- **Verification (BH p150a):** header-only + JIT kernel change (no `build_metal.sh` rebuild).
  `test_mcast_pipe.py` **39/39 PASS** — the green re-confirm of the materialization, exercising the
  no-loopback path across `n_iters=8` (proves ctor-set-once VALID + no per-send set holds across
  iterations), loopback (`test_f3_loopback`), the degenerate local-copy collapse, NoC1 corner order,
  and pre_handshake. No provisional dual-paths (none re-decided).
- **Hand-off:** re-invoke `apply-dm-helper helper_design/mcast_pipe/` → Tier-0 remigrates the
  stale@v5 kernels to v6 first (the matmul in0 sender's own-flag consumption is the in-context confirm
  of D1 — it must match the raw set-once pattern), then resumes the pending backlog. No manual fleet
  re-run.

---

## Round 7 — topology survey: CHAIN cross-id-relay GAP made explicit (2026-06-20)

- **Trigger:** `tune-dm-helper feedback-2.txt` — one claim + three deliverables: the `Pipe` is a STAR
  primitive (one shared `data_ready` sem id, A5 `set_multicast` of the sender's OWN cell, src==dst);
  the CHAIN / store-and-forward family needs a **cross-id relay** (`Semaphore::relay_multicast`,
  `noc_semaphore.h:192`, src sem ≠ dst sem) the Pipe cannot express; the gap was captured only
  *implicitly* (folded into the F2=FLAG tag) and never surfaced as a first-class capability gap.
  Deliverables: a topology matrix with SUPPORTED/GAP/OOS per cell; an explicit blocker line in
  `migration_audit/transformer_sdpa.md`; a capability note in `proposed_helpers.md`.
- **Re-entry routing (batched, upstream-first):** I1 `relay_multicast` is a missing primitive →
  **Step A**; I2 chain "mutable-doorbell → write-once `valid_sem`" hazard → **Step B**; I3 SDPA-audit
  blocker line → **Step C**; I4 topology survey + matrix → **Step ★ (Step D)**; I5 `proposed_helpers`
  capability note → **Step F**. Leftmost = A → one forward pass **A → B → C → D → F**.
- **Step E (bake-off) = NO-OP, no device.** relay buys **no perf** for the star (only avoids one local
  L1 `set()` store, negligible vs the byte-identical NoC mcast). relay-vs-`set_multicast` is **not a
  style fork** — it is forced by the chain topology (cross-id mandatory there, `ASSERT`-impossible for
  star). No new matrix cell, no variant to measure; coverage/perf maps stand.
- **Step G (materialize) = NO-OP, no version bump.** Chain family stays **DEFERRED** (ask was to make
  the gap explicit, not implement relay). Helper code unchanged → `MCAST_PIPE_API_VERSION` stays **6**.
  No fleet remigration owed.
- **Grounding (confirmed in code, not asserted):** `Semaphore::relay_multicast` exists at
  `noc_semaphore.h:192` with `ASSERT(local_l1_addr_ != dst_sem.local_l1_addr_)`; chain_link.hpp inits a
  write-once `valid_sem` to VALID (L140-143) and relays it into the next link's `receiver_sem` (L232);
  the current `SenderPipe` only does A5 same-cell `set_multicast` → structurally cannot relay.
- **Decisions:**
  - **A1 — contracted A5′ `relay_multicast`** (cross-id, src≠dst) as distinct from A5 (src==dst).
  - **B1 — catalogued H12 / INV12** (mutable doorbell can't be the chain broadcast source → separate
    write-once `valid_sem`; topology-forced INVARIANT, not a fork).
  - **C1 — blocker #5 added** to `transformer_sdpa.md` (cross-id relay GAP = root blocker for the
    reader_interleaved / exp_ring_joint_reader refactors).
  - **D1 — topology matrix** (`design/api_feasibility.md` Step ★ Round-7 addendum): T1 STAR=SUPPORTED,
    T2 CHAIN=GAP, T3 RING=GAP+OOS, T4 FABRIC=OOS, T5 fan-in=OOS; fine matrix over F1×handshake×
    loopback×pre_handshake.
  - **F1 — capability-gap note** added to `proposed_helpers.md` header (STAR-only; chain=GAP, deferred;
    future close likely a `RelayPipe`/forwarding-link face).
- **API before == API after: version 6 (UNCHANGED).** Paper-only re-entry; no migrated kernel goes
  stale; nothing owed to `apply-dm-helper`.
- **Artifacts touched:** `design/primitive_contracts.md` (A5′ + PRIMITIVES line), `design/hazards_catalog.md`
  (H12/INV12), `migration_audit/transformer_sdpa.md` (blocker #5), `design/api_feasibility.md` (Step ★
  Round-7 addendum + topology matrix), `proposed_helpers.md` (capability note), this changelog.
- **Verification:** none — documentation-only round (no helper edit, no device).

---

## Round 8 — consumer-sem optionality + arg reorder + 3 implementation fixes (2026-06-20)

- **Trigger:** `tune-dm-helper feedback-3.txt` — four claims: (1) add an `ASSERT` that `NOC_ID` matches
  the `Noc` the `SenderPipe` runs on (and review for other needed asserts); (2) `CONSUMER_READY_SEM_ID`
  shouldn't have to be passed when `PRE_HANDSHAKE=false` — reorder the args meaningfully; (3)
  `sender_in_rect` shouldn't be recomputed per `send()` — compute it in the ctor; (4) hoist
  `async_writes_flushed()` out of the `fence_()` `if constexpr` so the `else` disappears.
- **Re-entry routing (batched, upstream-first):** item 2 → **Step D** (signature / param-order /
  optionality = a contract change); items 1, 3, 4 → **Step G** (materialization invariants — enforce an
  already-stated precondition, ctor-precompute, internal refactor; no contract change). Leftmost = D →
  one forward pass **D → E → F → G**. No mooting, no conflicts (item 2 doesn't remove the subject of
  1/3/4). **Step E was a re-confirm no-op** (item 2 touches no style fork and adds no matrix cell —
  the same `wait`/`up` run under the same `if constexpr (PRE_HANDSHAKE)` guard) — **no device bake-off.**
- **Item-2 design (user pick — keep the named knob, make the sem optional, push the rarest knob last):**
  `CONSUMER_READY_SEM_ID` became a **trailing param defaulted to `UNUSED_SEM_ID`** (a reserved sentinel
  `0xFFFFFFFF`), guarded by `static_assert(!PRE_HANDSHAKE || CONSUMER_READY_SEM_ID != UNUSED_SEM_ID)`.
  `PRE_HANDSHAKE` moved **before** the sem (gate-then-resource); `DATA_READY_SIGNAL` moved to **last**
  (its `Counter` arm is the rarest/most-defaulted knob). Confirmed in-scope (invariant 5): two migrated
  call sites already use `PRE_HANDSHAKE=false` — ln-sharded `phase1_pipe`
  (`reader_mcast_sender_unary_sharded_ln.cpp`) and conv-WS `act_mcast_pipe`
  (`activation_reader_width_sharded.cpp`) — both were forced to pass a consumer sem the Pipe ignores.
- **Decisions:**
  - **D1 (item 2) — SenderPipe args** `<NOC_ID, DATA_READY_SEM_ID, NUM_ACTIVE_RECEIVER_CORES,
    PRE_HANDSHAKE=true, CONSUMER_READY_SEM_ID=UNUSED_SEM_ID, DATA_READY_SIGNAL=Flag>`;
    **ReceiverPipe args** `<DATA_READY_SEM_ID, PRE_HANDSHAKE=true, CONSUMER_READY_SEM_ID=UNUSED_SEM_ID,
    DATA_READY_SIGNAL=Flag>`. Both carry the `static_assert`. Side effect (improvement): the all-default
    `SenderPipe<NOC,DR,NUM>` now fails the assert, so a control-only sender (topk `send_signal`, which
    never gates) must declare `PRE_HANDSHAKE=false` honestly.
  - **G1 (item 1) — NoC-mismatch assert.** `ASSERT(noc_.get_noc_id() == NOC_ID)` in the SenderPipe ctor
    (only meaningful under `--dev`; the routing corners + `my_x/my_y` are baked for `NOC_ID`). Reviewed
    for other asserts: the `McastRect<NOC_ID>` ctor-arg type already forces rect/sender NoC agreement at
    compile time, and the new `static_assert` covers the handshake/sem coupling — no further runtime
    assert added.
  - **G2 (item 3) — `sender_in_rect` precomputed.** The method `sender_in_rect_()` is deleted; the ctor
    computes a `bool in_rect_` once (my coords + rect both fixed at construction). `send()` uses
    `in_rect_ && src_l1 != dst_l1` (only the src/dst aliasing varies per send).
  - **G3 (item 4) — flush hoisted.** `fence_()` now calls `async_writes_flushed()` unconditionally, then
    adds `async_atomic_barrier()` only on the Counter path. The `else` is gone.
- **API before (version 6):** `SenderPipe<NOC_ID, DATA_READY_SEM_ID, CONSUMER_READY_SEM_ID,
  NUM_ACTIVE_RECEIVER_CORES, DataReadySignal=Flag, PRE_HANDSHAKE=true>`,
  `ReceiverPipe<DATA_READY_SEM_ID, CONSUMER_READY_SEM_ID, DataReadySignal=Flag, PRE_HANDSHAKE=true>`.
- **API after (version 7):** signatures in D1 above. `McastRect<NOC_ID=noc_index>`, `send`/`receive`/
  `send_signal`/`receive_signal` bodies unchanged.
- **`MCAST_PIPE_API_VERSION` 6 → 7** (caller-facing: reordered template args + the now-optional
  consumer sem → every migrated SenderPipe/ReceiverPipe call site is rewritten). All Round-6 migrated
  kernels are now **stale@v6**. (Items 1/3/4 are internal-only and would not bump on their own.)
- **Artifacts touched:** `design/api_feasibility.md` (Round-8 addendum), `design/style_bakeoff.md` (E no-op note),
  `proposed_helpers.md` (header), this changelog, `mcast_pipe.hpp` (materialized: version, sentinel,
  both templates, both static_asserts, ctor assert + `in_rect_` precompute, `fence_` hoist, deleted
  `sender_in_rect_()`), the 3 unit-test kernels (`pipe_sender`/`pipe_receiver`/`pipe_f3_sender`).
- **Verification (BH p150a):** header-only + JIT kernel change (no `build_metal.sh` rebuild).
  `test_mcast_pipe.py` **39/39 PASS** — `test_smoke` (compile gate, handshake sender), `test_coverage`
  (flag+counter × rects × n_iters × payloads), `test_noc1_sender_corner_order`, `test_pre_handshake`
  (CR-provided handshake arm), `test_f3_loopback` + `test_f3_degenerate` (no-handshake arm with CR
  **omitted** — proves the new trailing-default path and the static_assert accepts it). No hangs. No
  provisional dual-paths (none re-decided).
- **Hand-off:** re-invoke `apply-dm-helper helper_design/mcast_pipe/` → Tier-0 remigrates the stale@v6
  kernels to v7 first (positional template args shifted; the two `PRE_HANDSHAKE=false` sites can now
  drop their dead consumer-sem arg), then resumes the pending backlog. No manual fleet re-run.

## Round 9 — Flag-path per-send VALID re-assert (M12b), rotating-role STAR fix (2026-06-20)

- **Trigger:** `tune-dm-helper feedback-4.txt` — one claim: restore the sender's per-send `set(VALID)`
  on the Flag path that Round 6 removed. Device-confirmed root cause + A/B bisect (WH, 2026-06-20) on
  `reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp`, test
  `test_matmul_2d_multiple_output_blocks_per_core[...transpose_mcast=True...in0_sharded=True-grid_size=(8,4)...b=1]`:
  CONTROL = v7 unmodified → **HANG**; TREATMENT = v7 + one line `set(VALID)` before the flag send → **PASS**.
- **Re-entry routing:** leftmost contradicted artifact = **Step B**. The root cause is hazard **H12**
  (a mutable doorbell can't be the broadcast source), already in the catalog but scoped to the CHAIN
  topology only. feedback-4 proves H12 **also fires for the rotating-role STAR** (a core that is both a
  sender and a receiver on the SAME shared `data_ready` cell — its receiver turn drives the cell INVALID,
  so the Round-6 ctor-once VALID is stale by its next sender turn), AND that the STAR has a **cheaper
  mitigation the catalog never listed** (re-assert VALID per send — the star's source IS its dest cell,
  `src==dst`, so it can re-store without the chain's cross-id relay). That is "a valid mitigation the
  catalog doesn't list" → Step B. One forward pass **B → C → D → E → F → G**, all downstream steps cheap
  re-confirms (no new device bake-off).
- **Decision — M12b (DOMINANT single path, coverage-decided, NOT a fork/knob):** the Flag-path
  `signal_ready_` re-asserts `data_ready_.set(VALID)` before the flag `set_multicast`, **unconditionally,
  not gated behind a predicate.** It covers BOTH cells: required for the rotating-role STAR (hang without
  it), a redundant no-op store for the pure STAR (its cell is never clobbered). Counter path untouched
  (monotone — no level flag, no source cell to refresh). Reverses Round-6 D1 with a corrected rationale:
  Round 6 dropped the set as "redundant for a STAR sender" — true for a *pure* STAR, but the helper also
  serves *rotating* senders whose data-ready cell doubles as a clear-after-wait receiver cell.
- **Re-decide, not re-measure (Step E):** the device evidence is feedback-4's A/B; no `bakeoff_*` matrix
  re-run. E.4 case 1 (DOMINANT). Recorded in `design/style_bakeoff.md` §Round-9.
- **R6 consequence:** the rotating-role STAR is now **migratable** with two Pipes (`SenderPipe` +
  per-round `ReceiverPipe` on the shared cell, `receive(sx,sy)` for the rotating coord). The earlier
  "R6 confirmed hard / same-core sender+receiver hangs" verdict was a *symptom* of the Round-6 bug, not
  infeasibility. (CHAIN gap unchanged — it still needs INV12 cross-id relay; source ≠ dest there.)
- **API before/after (version 7):** signatures UNCHANGED — the edit is internal to
  `SenderPipe::signal_ready_()`/`send()`. No template arg, knob, face, or count semantics changed.
- **`MCAST_PIPE_API_VERSION` 7 → 7 (NO BUMP).** Caller-facing API is identical; an internal `send()`-body
  change does not bump (G.4). No migrated call site is made stale; no fleet remigration is owed *by the
  version key*. (block_sharded is separately quarantined and recoverable — see hand-off.)
- **Artifacts touched:** `design/hazards_catalog.md` (H12 amendment + M12b), `migration_audit/_SUMMARY.md` +
  `migration_audit/matmul.md` (R6 now migratable), `design/api_feasibility.md` (Round-9 addendum, no change),
  `design/style_bakeoff.md` (Round-9 re-decide), `proposed_helpers.md` (header + baked-in-choices table + R6
  defer note), this changelog, `mcast_pipe.hpp` (materialized: per-send `set(VALID)` on the Flag path in
  `signal_ready_`, ctor + header comments corrected; version define left at 7), and the unit test
  (new `pipe_rotating.cpp` kernel + `test_rotating_role` 6-cell parametrization).
- **Verification (WH, this run):** header-only + JIT kernel change (no `build_metal.sh` rebuild).
  `test_mcast_pipe.py` **45/45 PASS** (39 prior + 6 new `test_rotating_role`). Regression-guard proof:
  temporarily reverting the `set(VALID)` line makes `test_rotating_role[n_iters=2-payload_tiles=1]`
  **HANG** (dispatch timeout, triage shows both `pipe_rotating.cpp` cores stuck) — so the new cell
  genuinely catches a regression back to ctor-once-VALID. Line restored; full suite green.
- **Provisional items:** none. M12b is coverage-decided (correctness), not a micro-bench dual-path.
- **Hand-off:** re-invoke `apply-dm-helper helper_design/mcast_pipe/`. Note: because the API version did
  NOT change, no Tier-0 staleness sweep is triggered by the version key. The work owed is specifically to
  **lift `block_sharded` out of quarantine** — its v7 call site is recoverable from commit `fa561f3b584`
  (raw revert currently at HEAD); with M12b in the helper it now passes. apply-dm-helper re-verifies it
  against its mapped test and flips the ledger entry `quarantined → migrated@v7`.

---

## Round 10 — split the recipient count (D2) + rect-derived fan-out (D1 half) (2026-06-20)

- **Trigger:** `tune-dm-helper` feedback round — close design-gap **D2** (report.md): the v7
  `NUM_ACTIVE_RECEIVER_CORES` template param served THREE jobs at once — the consumer-ack handshake wait
  (`send` :194), the data mcast `num_dests` (:204), and the signal mcast `num_dests` (:217). The fan-out
  (204/217) is the cores the broadcast physically lands on; the handshake (194) is only the *active*
  receivers that ack. They diverge whenever the mcast box holds inactive/noop cores (conv-WS,
  dram-sharded, conv-1D-weights), so those kernels stayed raw/deferred. Plus the PERF ask: precompute all
  constants in the ctor; make `send()` do no arithmetic.
- **Re-entry routing (Step D, leftmost):** the claim changes *what a count means* and splits one param
  into two — a contract change, no measurement disputed. One forward pass **D → E → F → G**. **Step E was
  a re-decide NO-OP, no device** (the split touches none of the four style forks — flush/barrier,
  flag/counter, linked/unlinked, EXCLUDE/INCLUDE; the conv-1D-weights hang it unblocks was already an
  A/B-confirmed coverage fact, report.md D2, not a perf measurement).
- **Decisions:**
  - **A — fan-out from geometry.** Re-added `McastRect::area()` = `(xhi-xlo+1)*(yhi-ylo+1)` on the
    normalized corners — **count use only, NOT loopback inference** (loopback stays the Round-3 rule
    `in_rect_ && src!=dst`). `SenderPipe` derives `num_dests_excl_ = area-(in_rect?1:0)` and
    `num_dests_incl_ = +1`. Because the rect corners are runtime, `area()` is runtime → **runtime fan-out
    for free (resolves D1's fan-out half).**
  - **B — handshake gets its own count.** New runtime ctor arg `consumer_ack_count`, **defaulting to the
    sentinel `ACK_EQUALS_FANOUT`** (= the derived EXCLUDE fan-out). Dense callers omit it; a divergent
    caller (conv-WS: `num_mcast_cores-1`; dram-sharded: `num_dram_banks`; conv-1D-weights: `total_active-1`)
    passes its own smaller count.
  - **PERF — ctor precompute.** The ctor caches `in_rect_`, `num_dests_excl_`, `num_dests_incl_`,
    `degenerate_(=excl==0)`, `ack_count_`. `send()` does only: degenerate→local-copy guard;
    `wait(ack_count_)`; `loopback = in_rect_ && src!=dst`; **branch-select** `mcast_dests = loopback ?
    num_dests_incl_ : num_dests_excl_` (no arithmetic); issue + fence.
  - **Proven invariant — `num_dests == area ± source`** (device-grounded this round across all 10 migrated
    senders, including the conv-WS case once feared a counterexample: the factory sets
    `num_reader_cores = all_cores.bounding_box().size() = area`, so its INCLUDE fan-out IS the area; what
    diverges there is the *ack* count — exactly the thing the split decouples). Consequence: **every dense
    call site re-migrates by simply dropping its count arg** — the rect carries the fan-out and the ack
    default. The explicit-ack arm's first consumers are the deferred divergence kernels this round
    unblocks (not a dead knob).
- **Scope guard:** D3–D9 untouched. **Per-send-varying ack is OUT** (`ack_count_` is ctor-cached; the
  sort coordinator's start-vs-substage ack varies per send — not covered, and D3-blocked anyway). Recorded.
- **API before (version 7):** `SenderPipe<NOC_ID, DATA_READY_SEM_ID, NUM_ACTIVE_RECEIVER_CORES,
  PRE_HANDSHAKE=true, CONSUMER_READY_SEM_ID=UNUSED, DataReadySignal=Flag>(noc, McastRect<>{...})`.
- **API after (version 8):** `SenderPipe<NOC_ID, DATA_READY_SEM_ID, PRE_HANDSHAKE=true,
  CONSUMER_READY_SEM_ID=UNUSED, DataReadySignal=Flag>(noc, McastRect<>{...}, consumer_ack_count =
  ACK_EQUALS_FANOUT)`. `McastRect` gains `area()`. `ReceiverPipe` UNCHANGED.
- **`MCAST_PIPE_API_VERSION` 7 → 8** (caller-facing: removed template param + added ctor arg + re-added
  `area()` → every migrated `SenderPipe` site is rewritten). All Round-9 migrated kernels are now
  **stale@v7**.
- **Artifacts touched:** `design/api_feasibility.md` (Round-10 addendum + invariant table), `design/style_bakeoff.md`
  (E re-decide no-op), `proposed_helpers.md` (header), this changelog, `mcast_pipe.hpp` (materialized:
  `area()`, `ACK_EQUALS_FANOUT` sentinel, SenderPipe template/ctor/send/send_signal/members), the three
  sender unit-test kernels (`pipe_sender`/`pipe_f3_sender`/`pipe_rotating`) + `test_mcast_pipe.py`
  (consumer-ack slot + two new cases).
- **Verification (Wormhole b0, this run):** header-only + JIT kernel change (no `build_metal.sh`
  rebuild). `test_mcast_pipe.py` **50/50 PASS** (45 prior + 3 `test_runtime_fanout` + 2 `test_split_count`).
  **Regression-guard proof:** temporarily passing the dense default ack (`ACK_EQUALS_FANOUT` → fan-out=4)
  in `test_split_count` makes the sender wait 4 acks while only 2 arrive → **HANG** (operation-timeout
  triage dispatched), so the new split-count cell genuinely catches the round-1 conv-1D-weights regression;
  explicit ack=2 restored, full suite green.
- **Provisional items:** none. The split is contract/coverage-decided (the divergence is a correctness
  fact), not a micro-bench dual-path.
- **Hand-off:** re-invoke `apply-dm-helper helper_design/mcast_pipe/`. The version bump (7→8) triggers a
  Tier-0 staleness sweep: every kernel with `migrated_api_version < 8` is remigrated first — for the
  dense sites this is a pure deletion of the count arg (the rect now carries it). Then apply-dm-helper can
  newly migrate the D2 divergence kernels (conv-WS-with-handshake, dram-sharded, conv-1D-weights) that the
  split unblocks, passing their explicit `consumer_ack_count`.
