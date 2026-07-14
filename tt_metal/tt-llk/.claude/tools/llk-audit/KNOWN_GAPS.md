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
- **Scope:** opt-in `--full-jit` tier only.

### L3 — no committed golden baseline / real-tree count assertion
`out/` is gitignored, so there is no committed golden fact base / per-arch count manifest and no
automated real-tree count assertion. A real recall regression (an extractor edit, a `registry.py`
change, or a tt-llk header refactor dropping e.g. mmio 91→60) passes all unit tests and leaves no
artifact to diff.
- **Risk:** CAP-REDUCTION — a silent recall drop.
- **Fix:** commit a `ground_truth.json` (per-arch, per-checker counts + sub-buckets) and add a
  CI/test step asserting `run.sh` output equals it; treat any intended change as a manifest edit.

### L4 — template-dependent cfg/mmio writes are surfaced but not classified
A template-dependent cfg/mmio write is emitted as `unresolved` (visible, never silently dropped) but
is not resolved to a config word; a full fix needs template-instantiation visiting in the extractor.
- **Risk:** partial recall (surfaced, not a silent miss).
- **Status:** deferred as risky.

### X1 — object-method `async_read_with_state` not recalled by `noc_is_read`
`noc_is_read` recalls the whole free-function `noc_async_read*` family by prefix, but its
object-method branch matches only the exact name `async_read` — so the object form
`noc.async_read_with_state<...>(...)` is not recalled. `noc-read-barrier` gates on `noc_is_read`, so a
consumed-before-barrier hazard expressed via the object API produces no finding.
- **Risk:** CAP-REDUCTION — a kernel-tier `noc-read-barrier` false-negative over ~19 ttnn object-API
  reader kernels (matmul dataflow).
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
(`async_read`, …) that don't start with `noc_`, and `noc_op_of` returns non-None only for write-flush
methods — so an object-API pure-reader Blackhole kernel whose poll pointer is not `get_semaphore`
provenance reads `has_noc == False` and its poll is dropped.
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
- **Live today:** none (the only `mop_cfg` site is a non-reconfig MOP template).
- **Fix hazard:** **CAP-REDUCTION TRAP** — a genuine `reinterpret_cast<volatile …>`-to-cfg write in a
  reconfig fn is *also* `mmio_ptr`, so blunt removal of `mmio_ptr` creates a false-negative. A correct
  fix must distinguish mop- vs cfg-provenance `mmio_ptr`.

## How to add an entry
Append a `### <id> — <one-line title>` block with **Risk** (CAP-REDUCTION / FALSE-FLAG), **Live
today**, **Fix**, and any **Fix hazard**. Link the relevant checker `blind_spots` rather than
restating it. When you fix one, delete its entry in the same change. This file — not a commit message
or a memory note — is the single home for deferred gaps.
