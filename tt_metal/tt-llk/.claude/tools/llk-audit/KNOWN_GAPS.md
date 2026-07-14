<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->
# llk-audit — Known limitations / deferred gaps

The single, canonical list of **known, deliberately-deferred** gaps in the `llk-audit` recall
tool. **Append any new deferred/latent gap HERE** — do not scatter it across commit messages or
duplicate it elsewhere. Per-checker caveats still live in each checker's `blind_spots` field
(`checks/*.py`); this file tracks the cross-cutting *deferred* items and links to the relevant
`blind_spots` where one exists.

A gap listed here is **real but consciously not fixed** — usually because the obvious fix would
*reduce recall* (add false-negatives) or *add false-positives*. It is **not** a refuted / non-issue.
Do not re-file these as new findings; do fix one when the right approach is found, then remove it.

Risk legend — **CAP-REDUCTION**: the tool could miss a real hazard (false-negative). **FALSE-FLAG**:
the tool could over-report (false-positive). For a recall tool, cap-reduction is the more serious
class, so several fixes below are deferred precisely because a naive fix trades one for the other.

## Deferred items

### L1 — compound-assign cfg RMW modeled as a full-word clobber
The extractor records the assignment operator (`f.op`: `=`, `|=`, …) on a `pointer_write`, but
`registry.py` never consults it, so a read-modify-write `cfg[FIELD_ADDR32] |= v` is classified as a
full-word write.
- **Risk:** FALSE-FLAG — a spurious `INTRA_THREAD_CLOBBER` (a `|=` preserves sibling bits).
- **Live today:** none (0 `cfg[]|=` sites in tt-llk).
- **Fix hazard:** gating "full word" on `op == "="` would demote a genuinely *torn* cross-thread
  non-atomic `|=` from `POTENTIAL_CLOBBER` → **CAP-REDUCTION**. A correct fix must split the
  intra-clobber path (op-sensitive) from the cross-thread safety path (op-insensitive).

### L2 — kernel-tier: a parse-failed non-kernel TU can read as coverage-clean
Kernel tier only (`--full-jit`). In `kernel_tier/capture.py`, a translation unit that parses with
errors (`parse_errors > 0`) yet yields 0 kernel-surface facts is logged `ok(...)nonkernel`; because
the status leads with `ok`, `bootstrap.sh`'s anchored HOLE grep never counts it, so the parse hole
does not reach the audit-JSON `degraded`.
- **Risk:** CAP-REDUCTION — a kernel that *should* have contributed facts but failed to parse can
  read as coverage-clean to a JSON consumer.
- **Live today:** none observed (opt-in `--full-jit` tier only; not exercised on the default path).
- **Fix:** in `tu_ledger_status`, give a `parse_errors>0 && 0-kept` TU a HOLE-leading status so
  bootstrap's grep counts it.

### L3 — no committed golden baseline / real-tree count assertion
`out/` is gitignored, so there is no committed golden fact base / per-arch count manifest and no
automated real-tree count assertion. A real recall regression (an extractor edit, a `registry.py`
change, or a tt-llk header refactor dropping e.g. mmio 91→60) passes all unit tests and leaves no
artifact to diff.
- **Risk:** CAP-REDUCTION — a silent recall drop.
- **Live today:** N/A — structural (the guard is simply absent), not a per-site latent case.
- **Fix:** commit a `ground_truth.json` (per-arch, per-checker counts + sub-buckets) and add a
  CI/test step asserting `run.sh` output equals it; treat any intended change as a manifest edit.

### L4 — template-dependent cfg/mmio writes reach no checker (silent miss)
A cfg/mmio write whose base pointer is template-dependent gets `provenance_kind="unresolved"` from the
extractor, so `classify_write` returns `(None, None)` and ALL THREE pointer_write consumers drop it
with no Finding: mmio-race (`if kind:`), reconfig-stall (`kind in CFG_WRITE_KINDS`), and cfg-word
(`is_cfg_looking_unrecognized` gates on `provenance_kind in ("call","var","cast")`, excluding
`"unresolved"`). The write stays in `facts.<arch>.jsonl` but produces no audit Finding — so to the
skill/LLM worklist it is a SILENT miss. (Distinct from a RECOGNIZED cfg write with an unresolvable
INDEX, e.g. `cfg[runtime_var]=`, which DOES emit a `hint=UNRESOLVED` Finding.)
- **Risk:** CAP-REDUCTION (silent — no Finding emitted).
- **Live today:** none — every `unresolved` pointer_write over tt-llk is (a) a local staging-struct
  write (`config.val[i]=` / `tile_descriptor.val[i]=`; QSR `alu_config.val[i]=`), (b) a RISC↔RISC
  mailbox-FIFO write (`mailbox_base[thread][i]=`), or (c) a QSR buffer-descriptor-table /
  TRISC-sync-enable write (`bd_table[…].words[i]=` / `t6dbg->TENSIX_TRISC_SYNC[i]=`, the
  `set_ttsync_enables` store) — NONE a direct unit-sampled cfg-pointer write, so the `(None,None)` is
  not a real cfg write being dropped (a real cfg write is separately routed through a recognized
  `TTI_WRCFG`/`REG2FLOP`/`cfg[]=`). Reproduced counts: WH 19 (13 `config.val` + 5 `tile_descriptor.val`
  + 1 mailbox), BH 17 (11 + 5 + 1), QSR 6 (2 `alu_config.val` + 2 `t6dbg`-TRISC_SYNC + 1 `bd_table`
  + 1 mailbox).
- **Fix:** extractor template-instantiation visiting to resolve the base word; deferred as risky.

### L5 — diff-scope and the degrade-guard match changed files by BASENAME
`--changed` mode only. `cli.py`'s `scope_to_changed` and guard-3 ("changed file(s) contributed NO
facts") both key on `os.path.basename(...)` — because every evidence line is formatted
`basename:line …` and scope matching must anchor on that prefix. If a changed `dirA/foo.h` fails to
parse while a *different* `dirB/foo.h` yields facts, the shared basename `foo.h` makes the
degraded-note suppressed AND can let a `dirB/foo.h` finding surface in a scope meant for
`dirA/foo.h` — a not-analyzed changed file reads as analyzed.
- **Risk:** CAP-REDUCTION — a false-all-clear-adjacent path: a not-analyzed changed file reads clean.
- **Live today:** none — 0 duplicate `.h` basenames within either arch's audited set (`llk_lib` +
  `common/inc`, incl. `common/inc/sfpu/`) on WH/BH/QSR, and `run.sh` restricts `--changed` to the
  *current* arch tree, so a cross-arch basename collision cannot occur in one run.
- **Fix:** key scoping on the full repo-relative path — which requires re-encoding every checker's
  evidence line from `basename:line` to `path:line` (scope_to_changed anchors on that prefix).
  Deferred: broad churn across all checkers + their tests for a dormant case.
- **Fix hazard:** none of substance — a path-keyed rewrite is behavior-preserving wherever basenames
  are unique (today, everywhere), so there is no live divergence to regression-test it against.

### L6 — extractor emits no `pointer_write`/`pointer_read` for MEMBER-ACCESS writes (`ptr->field=`)
`extractor/llk_extract.cpp` matches an assignment LHS only as `ArraySubscriptExpr` (`ptr[i]=`) or
`UnaryOperator`/`UO_Deref` (`*ptr=`); a `MemberExpr` LHS (`ptr->field = v`) matches neither, so NO fact
is emitted — systemic across BOTH `VisitBinaryOperator` (writes) AND `VisitImplicitCastExpr` (the
`pointer_read` poll source). A cfg/MMIO write via a typed-struct pointer member access is thus
invisible to mmio-race / cfg-word / reconfig, and a `while(flag->field)` volatile member poll is
invisible to noc-l1-invalidate. Worse than L4: the fact is never emitted, so it is invisible even to a
raw `facts.<arch>.jsonl` inspection (L4's writes ARE emitted, just unrouted).
- **Risk:** CAP-REDUCTION — a silent miss (no fact, no finding).
- **Live today:** QSR-only recall ASYMMETRY, not a live race. `ckernel_(unpack_)template::program`
  program the MOP via `reinterpret_cast<mop_config_regs_t*>(MOP_CFG_BASE); mop_cfg->FIELD = …`
  (~30 sites) — all dropped; WH/BH use the `mop_cfg[i]=` array form and ARE captured (17 mmio-race
  NO_LOCAL_ORDERING each; 0 on QSR). Those QSR writes are ordered by the **MOP-config-bank HW
  backpressure / `mop_sync()`** (Confluence "MOP CFG double buffering" 113017192; LLK `program()`
  comment "in use should block, so no mop_sync() needed"), **not** TTSync (Confluence 1340276980
  EXCEPTS `MOP_CFG` from the RQ) — so a captured candidate would be LLM-adjudicated safe exactly like
  the WH/BH ones. The only other member-access cfg write in tt-llk (QSR `ckernel_riscv_debug.h`
  `RISCV_DEBUG_REGS->…`) is in an uncompiled header, never in the fact base.
- **Fix:** add a `MemberExpr` LHS branch to both visitors, capturing the member name (member writes
  carry no numeric `index_text`, which an index-keyed checker path would need). Low-hazard (`mop_cfg`
  provenance resolves to the base cast → `mmio_ptr`; `program` is not a `RECONFIG_FN`, so no
  reconfig-stall false-flag) but nontrivial → deferred. Related: the QSR `AUTOTTSYNC_ORDERED` blanket
  tag (see mmio-race `blind_spots`) over-clears the excepted/non-CFG-GPR subset these MOP writes belong
  to — even if captured they'd need the exception-aware tagging, not a blanket auto-order.

### L7 — QSR `AUTOTTSYNC_ORDERED` blanket tag ignores TTSync's RQ exceptions + non-CFG/GPR MMIO
`mmio_race.py` converts every unguarded QSR write `NO_LOCAL_ORDERING → AUTOTTSYNC_ORDERED` (pre-cleared,
"not a race candidate"). QSR genuinely enables Auto TTSync (`set_ttsync_enables<TRACK_ALL>`,
`llk_math_common.h:32`), so this is CORRECT for a tracked CFG/GPR write with an RQ-tracked consumer. But
per Confluence "Every Conceivable TTSync Detail" 1340276980, the RQ EXCEPTS `RESOURCEDECL`, `MOP_CFG`,
`REPLAY(load=1)`, and post-load-replay instructions — a write consumed by one of those is NOT
auto-ordered — and TTSync tracks only CFG/GPR/TDMA, not other MMIO spaces (e.g. the replay unit
`replay_mmap[...]`). The blanket tag over-clears both.
- **Risk:** CAP-REDUCTION — pre-clears (marks non-candidate) a write TTSync does not actually order.
- **Live today:** 10 `replay_mmap[...]` writes in `start/finish_using_replay_mmio_load` (replay-unit
  MMIO, in a REPLAY-load context) are tagged `AUTOTTSYNC_ORDERED`. They are NOT a live race — ordered by
  an explicit `fence`+CSR chicken-bit + `wait_replay_idle()` + mutex (ref TEN-2139) — but the tag's
  *reasoning* is wrong (attributes safety to TTSync). Disclosed in mmio-race `blind_spots`.
- **Fix:** tag by consumer/target — auto-clear only a CFG/GPR write whose consumer is RQ-tracked;
  SURFACE (not pre-clear) a write consumed by MOP_CFG/REPLAY/RESOURCEDECL or targeting non-CFG/GPR MMIO.
- **Fix hazard:** FALSE-FLAG if narrowed bluntly — QSR MOP/replay writes are generally safe (bank
  backpressure / fence-mutex), so surfacing them all would over-flag; a correct fix needs consumer-type
  + target-space discrimination the tool does not have today → deferred.

### X1 — object-method `async_read_with_state` not recalled by `noc_is_read`
`noc_is_read` recalls the whole free-function `noc_async_read*` family by prefix, but its
object-method branch matches only the exact name `async_read` — so the object form
`noc.async_read_with_state<...>(...)` is not recalled. `noc-read-barrier` gates on `noc_is_read`, so a
consumed-before-barrier hazard expressed via the object API produces no finding.
- **Risk:** CAP-REDUCTION — a kernel-tier `noc-read-barrier` false-negative over ~26 object-API reader
  files (≈20 under ttnn/cpp, the rest test kernels) spanning matmul, concat, transpose, untilize,
  sdpa, pool, topk — only ~6 are matmul.
- **Live today:** none — every live site is batched-safe (`set_state → read → barrier → push`).
- **Fix:** broaden the object branch to `startswith("async_read")` minus barrier/flush forms, gated
  on the `Noc` receiver type (mirrors the free-fn branch). No object `async_read_set_state` exists to
  be wrongly admitted.
- **Fix hazard:** bounded FALSE-FLAG — the same already-accepted over-flag class as the free-fn
  branch. NB this *broadens the object branch*; it is distinct from the deliberate decision **not to
  narrow** the free-fn matcher (see `noc-read-barrier` `blind_spots`). Verify against a captured
  kernel base (batched-safe kernels must stay at 0) before landing.

### X2 — `noc-l1-invalidate` `has_noc` scope misses the object NoC API
`noc-l1-invalidate` scopes a candidate poll to a dataflow kernel via
`has_noc = any(name.startswith("noc_")) or any(noc_op_of(c))`. Object-API call names are bare methods
(`async_read`, …) that don't start with `noc_`, and `noc_op_of` recognizes an object method only if it
is a flush / full-barrier or a Semaphore signal — NOT an object-method READ (`async_read`) — so an
object-API pure-reader Blackhole kernel whose poll pointer is not `get_semaphore` provenance reads
`has_noc == False` and its poll is dropped.
- **Risk:** CAP-REDUCTION — a kernel-tier, Blackhole-only false-negative. Materially the same as the
  checker's own `blind_spots` ("a poll whose function has no NoC call is MISSED").
- **Live today:** none (candidate kernels carry a free `noc_` call or a `get_semaphore` poll).
- **Fix:** count object-form NoC calls (`recv_type == "Noc"`) in `has_noc` — a clean recall gain,
  negligible FP risk.

### X3 — `reconfig-stall` treats a MOP/cast `mmio_ptr` write as a reconfig cfg-write
`reconfig-stall`'s `is_cfg_write` accepts any `pointer_write` whose `classify_write` kind is in
`CFG_WRITE_KINDS`, which includes `mmio_ptr` (a `mop_cfg` variable, or a volatile cast to a
`cfg`/`mop`/`base` target). A MOP-buffer write in a reconfig-named function is not a unit-sampled
config register, so it would draw a spurious `NO_UNIT_DRAIN`.
- **Risk:** FALSE-FLAG.
- **Live today:** none — no `mmio_ptr` (`mop_cfg`/cast) write sits in a reconfig-named function.
  (The `mop_cfg` sites are `ckernel_template::program` and `ckernel_unpack_template::program` — both
  compile to `function="program"` in the facts — on **WH+BH only** (17 `mmio_ptr` writes each, via
  `mop_cfg = reinterpret_cast<…>(TENSIX_MOP_CFG_BASE); mop_cfg[i] = …`, i.e. ARRAY-SUBSCRIPT writes).
  QSR's `program` ALSO base-casts (`reinterpret_cast<mop_config_regs_t*>(MOP_CFG_BASE)`, its
  `instrn_buffer` param is `[[maybe_unused]]`) but writes via `mop_cfg->FIELD =` (a typed-struct MEMBER
  access), which the extractor does NOT emit as a `pointer_write` (see **L6**) — so QSR emits **0**
  `program` `mmio_ptr` writes. ("0 on QSR" = 0 *program-fn* `mmio_ptr`, not 0 overall — QSR has 72
  `mmio_ptr` writes elsewhere.) Plus the experimental BH pack helpers `_llk_pack_block_contiguous_` /
  `_llk_pack_fast_untilize_mop_patch_last_`. None are `RECONFIG_FN_SUBSTR` names, so `is_reconfig_fn`
  skips them.)
- **Fix:** give reconfig-stall its own kind set excluding `mmio_ptr` (distinguishing mop- from
  cfg-provenance — see Fix hazard).
- **Fix hazard:** **CAP-REDUCTION TRAP** — a genuine `reinterpret_cast<volatile …>`-to-cfg write in a
  reconfig fn is *also* `mmio_ptr`, so blunt removal of `mmio_ptr` creates a false-negative. A correct
  fix must distinguish mop- vs cfg-provenance `mmio_ptr`.

## How to add an entry
Append a `### <id> — <one-line title>` block with **Risk** (CAP-REDUCTION / FALSE-FLAG), **Live
today**, **Fix**, and any **Fix hazard**. Link the relevant checker `blind_spots` rather than
restating it. When you fix one, delete its entry in the same change. This file — not a commit message
or a memory note — is the single home for deferred gaps.
