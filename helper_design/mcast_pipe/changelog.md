# mcast `Pipe` — change log

Round-by-round record of the helper's API and rollout. Each round: trigger →
decisions → artifacts touched → device verification. Diff API states here when a new
feedback round lands.

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
- **Decisions** (see `round2_active_cores_plan.md`):
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
  **width-sharded activation sender REVERTED to raw** — un-inferable partial-box self-gather
  (see `loopback_inference_limitation.md`), raw test PASS. ✓
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
    The `LINK=false` arm was justified in `proposed_helpers.md`/`style_bakeoff.md` by *"a barrier is
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
- **Artifacts touched:** `api_feasibility.md` (Round-5 addendum), `style_bakeoff.md` (E no-op note),
  `proposed_helpers.md` (header), this changelog, `mcast_pipe.hpp` (materialized), the 3 unit-test kernels
  + `test_mcast_pipe.py` (ported to the new API).
- **Verification:** header-only + JIT kernel change (no `build_metal.sh` rebuild). `test_mcast_pipe.py`
  unit gate is the green re-confirm of the materialization. Provisional dual-paths: none new (F4 linking
  stayed baked-in; no fork re-decided).
- **Hand-off:** re-invoke `apply-dm-helper helper_design/mcast_pipe/` → Tier-0 remigrates the 13
  stale@v4 kernels to v5 first, then resumes the 46 pending. No manual fleet re-run.
