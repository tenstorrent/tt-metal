<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# llk-audit — deterministic recall engine for the LLK race audits

An AST-based tool that **exhaustively enumerates the KNOWN hazard patterns** for
several LLK race-audit classes and emits a candidate list the `/*-audit` skills
consume. It is an **augmentor, never a gate**: it finds every instance of the
patterns it can encode, but a tool cannot recognize *unknown* patterns, so its
output is advisory input to the LLM/human — which still owns the verdict and the
hunt for novel hazards.

```
┌─ extractor/  (C++ / Clang libTooling) ─ parse once, emit a semantics-free FACT BASE
│     llk_extract.cpp     functions · pointer-writes (+provenance) · calls · macro expansions
│
├─ llkaudit/   (Python) ─ classify the fact base into recall candidates
│     registry.py         ← THE table that maps LLK names/signatures to meaning (edit this)
│     factbase.py         load / merge / dedup / query
│     checks/*.py         one checker per hazard class (augmentors)
│     cli.py              run checks, emit one advisory JSON envelope
│
└─ run.sh      orchestrate: extract every header for an arch → run checks
```

## Why this split

The fragile, fidelity-critical work — parsing template/macro-heavy 3-arch
headers, tracing pointer provenance, recovering macro names+args the AST discards
(`TTI_STALLWAIT`, `p_stall::TRISC_CFG`, …) — needs libTooling and only has to be
gotten right **once**. The frequently-edited work — *which* names count as *what*
— is data in `registry.py`. **When an LLK signature changes, you edit one table
in `registry.py`; you rarely touch a checker and never the C++.**

## Checks (this build)

| Check | What it recalls | Hints |
|---|---|---|
| `mmio-race` | RISC MMIO cfg/GPR write vs. consuming Tensix instruction/MOP; is an applicable ordering primitive local? | `LOCALLY_ORDERED` / `NO_LOCAL_ORDERING` |
| `cfg-word-overlap` | fields sharing one 32-bit CONFIG word (per register file) written by ≥2 threads; intra-thread full-word clobber | `CROSS_THREAD_SHARED_WORD` / `INTRA_THREAD_CLOBBER` / `UNRESOLVED` |
| `semaphore-handshake` | mutex acquire/release imbalance; semaphore wait with no matching concrete init (emitted as a candidate, `safety: LOW_CONFIDENCE` when a generic init may cover it) | `MUTEX_IMBALANCE` / `WAIT_WITHOUT_INIT` |
| `reconfig-stall` | reconfig/uninit config write missing a unit-draining stall (walks every write; models unit re-arm) | `NO_UNIT_DRAIN` / `THCON_ONLY` / `DRAIN_REARMED` |
| `srcreg-bank` | SrcA/SrcB data-valid handshake control points; raw `SETDVALID` on Blackhole (ISA-unsupported) | `RAW_SETDVALID_BH` / `DVALID_SET` / `DVALID_CLEAR` |
| `mailbox-sync` | in-tree RISC↔RISC mailbox FIFO endpoints + writer↔reader pairing by directed channel | `PAIRED_CHANNEL` / `UNPAIRED_ENDPOINT` / `UNRESOLVED_ENDPOINT` |
| `cb-sync` † | circular-buffer reserve/push & wait/pop credit balance per CB (within a function) | `CB_RESERVE_PUSH_IMBALANCE` / `CB_WAIT_POP_IMBALANCE` |
| `noc-sync` † | NoC credit signal (`noc_semaphore_inc/set/mcast`) with no preceding write flush/barrier | `NOC_SIGNAL_NO_FLUSH` |

† `cb-sync` / `noc-sync` are committed and deterministic, but their surface is
JIT-compiled kernels OUTSIDE tt-llk — they only emit findings when fed a **kernel
fact base** (the on-request JIT capture; see *kernel tier* below). Over the tt-llk
fact base they are trivially empty (there are no cb/noc sites there).

Every finding is a **recall bucket, not a verdict**, and every check declares its
`blind_spots` in the output. `srcreg-bank` and `mailbox-sync` are deliberately
**narrow recallers**: `srcreg-bank` enumerates the dvalid control points and
flags the one mechanical ISA pattern (raw `SETDVALID` on BH) but does NOT model
the bank-flip lockstep verdict; `mailbox-sync` covers only the tiny IN-TREE
mailbox surface (all mailbox use outside tt-llk — the compute API plus the
hand-written `mailbox_write` in ttnn/models kernels, one-to-one channels and
fan-outs alike — is out-of-tree, audited by the skill's ttnn-widened grep) and
pairs statically — balance/symmetry/overflow/ordering stay with the skill.

**cfg-word `safety` annotation.** A `CROSS_THREAD_SHARED_WORD` finding is *always*
emitted (multi-thread access to a word is worth seeing even when race-safe — it
can be an ownership smell), with a `safety` sub-annotation, never a filter:
`SAFE_BY_MASKING` (all cross-thread writers are byte-atomic masked RMW on disjoint
bits — provably not a data race), `POTENTIAL_CLOBBER` (a full-word write, a
non-atomic software `cfg_rmw`, or overlapping bits — the LLM must check
value-invariance/ordering), or `UNKNOWN` (a field mask wasn't in cfg_defines).
The per-thread bit masks are in the detail so you can judge ownership.

**Diff-scoped mode.** `./run.sh <arch> --changed [BASE]` (BASE defaults to `main`)
reports only findings that TOUCH a changed LLK header — the anchor file *or* any
evidence line (so a shared word whose partner writer is in a changed file still
surfaces). The whole tree is still parsed for cross-file context; only output is
scoped. Use it for a PR-scoped audit.

### Coverage: 8 of 9 classes have a committed checker
Only `instruction-latency` has no committed checker (its surface is the SFPU files
that don't parse under clang, and its verdict needs an out-of-tree version-pinned
`sfpi-gcc` table → stays fully LLM-driven). The split of the other 8 by *where
their surface lives*:
- **In tt-llk (findings on a plain run):** `mmio-race`, `cfg-word-overlap`,
  `semaphore-handshake`, `reconfig-stall`, `srcreg-bank`, `mailbox-sync`.
  `srcreg-bank`/`mailbox-sync` are narrow recallers (control-point/endpoint
  inventory + the one mechanical ISA flag); their kernel-layer surface stays with
  the skills' ttnn-widened grep.
- **In JIT-compiled kernels outside tt-llk (`cb-sync` / `noc-sync`):** the checkers
  are committed and deterministic, but need a **kernel fact base** to yield
  findings. Producing that (the JIT capture) is the only fragile, off-main,
  **on-request** step — the checkers themselves are permanent and cost nothing to
  keep. `run.sh --full-jit` *runs* the capture if built; it never builds it, and
  degrades honestly when absent (each class's skill still LLM-audits the kernel
  surface meanwhile). See the *Full-audit kernel tier* runbook in `race-audit-all`.

## Build & run

```bash
./run.sh wormhole                  # or blackhole | quasar   [--checks a,b] [--changed [BASE]] [--full-jit] [out_dir]
                                   #   --changed scopes output to files changed vs BASE (default main)
                                   #   --full-jit ALSO runs the opt-in kernel tier (cb/noc/mailbox) IF built
                                   #     (it never builds it — that is on-request, off-main); degrades
                                   #     honestly + names the uncovered classes when the module is absent
                                   #   auto-builds the C++ extractor on first run
                                   #   (or if stale); needs Clang/LLVM >= 18 dev libs.
./run.sh --kernel-tier-status      # probe the kernel-tier capability ("available"/"unavailable")
extractor/build.sh                 # optional: build the extractor manually
python3 tests/test_checks.py       # hermetic unit tests (no clang/repo needed)
```

The `/*-audit` skills invoke `run.sh` in their "Recall preflight", so simply
invoking the skill triggers the one-time extractor build. If no suitable Clang
is present, the build fails gracefully and the skill proceeds with its manual
method (its preflight says: absence of the tool != "no findings").

`run.sh` writes `out/facts.<arch>.jsonl` (fact base) and `out/audit.<arch>.json`
(advisory findings). Feed `audit.<arch>.json` to the matching `/*-audit` skill as
its pre-enumerated worklist.

## Updating when signatures change

Open `registry.py` — it is organized by concept with an `EDIT HERE` banner:
- new cfg-pointer accessor → add to `CFG_POINTER_PRODUCERS`
- new MMIO write call → `MMIO_WRITE_CALLS`
- renamed/added instruction macro → the relevant `*_SUBSTR` list
- new drain/sync function → `DRAIN_CALLS`
- new reconfig function name → `RECONFIG_FN_SUBSTR`; new latched register → `LATCHED_FIELDS`
- new semaphore/mutex wrapper → `SEMAPHORE_CALLS`
- new dvalid op / renamed SETDVALID/CLEARDVALID → `SRCREG_DVALID_OPS`
- new mailbox primitive → `MAILBOX_FIFO_CALLS`
- new/renamed CB credit call → `CB_CALLS`
- new NoC semaphore/flush call → `NOC_SIGNAL_CALLS` / `NOC_WAIT_CALLS` / `NOC_FLUSH_CALLS`

Then `python3 tests/test_checks.py` to confirm nothing regressed.

## Coverage boundaries (explicit — no silent caps)

1. **Augmentor only.** No gate, no "safe" claim, always exits 0. Green = "no new
   *known-pattern* instance," never "no bug."
2. **Interprocedural linkage is the LLM's job.** `mmio-race` consumer/guard
   association is intra-function; a caller-supplied guard shows as
   `NO_LOCAL_ORDERING`.
3. **SFPU stubbed.** `sfpi.h`/`sfpi_classes.h` are stubbed so headers that merely
   include them parse; files that structurally use `sfpi::` types fail to parse
   and are counted in `parse_errors` / logged to `out/parse.log`.
4. **cfg-word-overlap** partitions `Config` vs `ThreadConfig` by the write
   instruction (`SETC16` → `ThreadConfig`, every other write → `Config`; THCON is
   a sub-range of `Config`, not its own file);
   fields that don't resolve to an ADDR32 are reported `UNRESOLVED`. Whether a
   shared word actually races (bit-disjoint masking, mutex/semaphore ordering,
   value-invariance) is deferred.
5. **Quasar** is validated. Its config-write idiom is `cfg_rmw(FIELD_RMW, …)`
   (the `_RMW` composite is captured at the preprocessor level so the field
   resolves). Its per-RISC **TTSync/AutoTTSync** HW-orders the RISC-write ->
   consuming-instruction direction (Confluence "Every Conceivable TTSync Detail",
   1340276980), so mmio-race reports those as `AUTOTTSYNC_ORDERED` rather than
   race candidates — it does NOT cover an MMIO *read* awaiting a multi-cycle
   result (`wait_*_idle`), nor the `EN_SUBDIVIDED` cross-unpacker corner.

## Validated against ground truth

- `mmio-race` reproduces the earlier hand-validated result (WH 91: 67/24,
  BH 79: 60/19 — one BH write is `NO_LOCAL_ORDERING` because its only guard
  follows a consumer, which the "guard must precede the first consumer" rule
  correctly refuses to credit), incl. `_llk_unpack_A_`→`LOCALLY_ORDERED` and the
  `THCON_SEC0_REG3_Base_address` writes→`NO_LOCAL_ORDERING`.
- `cfg-word-overlap` finds the known shared MAIN words (ALU-format words 0/1/2 incl.
  STACC_RELU on WH; 1/2 on BH) and suppresses the THCON/main same-index false alias.
- `reconfig-stall` flags `set_packer_strides` etc. and correctly allowlists the
  latched `program_packer_destination` (`L1_Dest_addr`); exercises `THCON_ONLY` on BH.
- `semaphore-handshake` sees all ops (17 post / 20 get / 4 init / balanced mutexes
  on WH) and correctly reports no mutex imbalance — after excluding wrapper defs +
  RAII. It emits the in-tree waits (e.g. `FPU_SFPU`, `UNPACK_TO_DEST`) as
  `WAIT_WITHOUT_INIT` candidates tagged `LOW_CONFIDENCE` (the generic
  `t6_sem(index)` init may cover them) rather than globally suppressing them —
  recall-biased, per the augmentor contract.
- `srcreg-bank` (BH) flags the raw `TTI_SETDVALID` sites — including the three real
  ones in `llk_math_eltwise_unary_datacopy.h` (ISA-unsupported on BH, a genuine
  recall win) — and recalls the `CLEARDVALID` control points; on WH/QSR the same
  `SETDVALID` is a plain `DVALID_SET` candidate (no BH flag).
- `mailbox-sync` (WH/BH) pairs the `MATH→UNPACK` dst_index channel (cmath write ↔
  cunpack read → `PAIRED_CHANNEL`) and marks the thread-agnostic `ckernel_debug.h`
  halt/unhalt endpoints `UNRESOLVED_ENDPOINT`; Quasar has no in-tree FIFO → 0.
- `cb-sync` / `noc-sync` are **unit-validated only** (synthetic fact bases:
  reserve/push & wait/pop imbalance, signal-without-preceding-flush) — they report
  0 over tt-llk (no cb/noc sites) and are end-to-end validated on real kernels only
  once the JIT capture (kernel fact base) is built.
- **Quasar**: all 122 `cfg_rmw` writes resolve and are each single-thread-owned
  (12 PACK-only, 7 UNPACK-only words) → 0 cross-thread shared words, matching the
  skill's per-engine-ownership conclusion; mmio-race's 169 unguarded writes are
  correctly `AUTOTTSYNC_ORDERED`; reconfig recall works via `cfg_rmw`.

The HW claims the checkers encode are grounded in the tt-isa-docs
(`BackendConfiguration.md`: the `Config` vs `ThreadConfig` split and that `SETC16`
alone writes `ThreadConfig`; `RMWCIB.md`: byte-atomic masked RMW of `Config`;
`STALLWAIT.md`: `TRISC_CFG` = condition C13, and the drain-unit condition bits;
`MemoryOrdering.md`: the store-then-store race). Known modeling limitations
(StateID banks, STALLWAIT block-mask coverage, interprocedural linkage) are
listed per check in `blind_spots`.
