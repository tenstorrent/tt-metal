---
name: race-audit-all
description: Run all nine LLK hazard audits (mmio-race, reconfig-stall, cfg-word-overlap, semaphore-handshake, mailbox-sync, dataflow-cb-sync, srcreg-bank-sync, noc-sync, instruction-latency) across four synchronization surfaces, and add a cross-class JOIN pass that catches emergent races no single audit can see — where one audit's verdict is "safe because <invariant owned by another audit>". Use for a full hazard sweep of an LLK change, or before merging anything touching config writes, reconfig/uninit, inter-thread/cross-core sync, the SrcA/SrcB-Dst data path, or hand-written instruction sequences.
user_invocable: true
---

# /race-audit-all — Orchestrated LLK race sweep + cross-class synthesis

## Purpose
The per-class audits each cover one hazard mechanism. Most real bugs are category-local and any one audit finds them. But some are **emergent**: each audit individually says SAFE because its safety *depends on an invariant that lives in another audit's domain*, and no single audit verifies the join. This skill runs all nine and adds a **JOIN pass** that discharges those cross-references — without ever losing what the individual audits found.

The nine sub-audits span **four synchronization surfaces** (not just cross-thread races — the suite also covers cross-core, RISC↔Tensix, and intra-thread micro-architectural hazards). Run each fully — see its own SKILL.md:
- **Cross-thread (shared backend state):**
  - `cfg-word-overlap-audit` — two threads write the same 32-bit CONFIG word.
  - `reconfig-stall-audit` — config rewrite without draining the consuming execution unit.
  - `semaphore-handshake-audit` — inter-thread semaphore/mutex protocol (incl. SEMINIT-vs-usage).
  - `srcreg-bank-sync-audit` — SrcA/SrcB `AllowedClient`+bank-flip handshake (unpacker↔Matrix Unit) and shared-once Dst/LReg overwrite.
- **RISC↔Tensix ordering:**
  - `mmio-race-audit` — RISC MMIO write vs Tensix-instruction/MOP/replay ordering; also `mop_sync`/`tensix_sync` drains (incl. OVER-SYNC/REDUNDANT perf findings).
  - `mailbox-sync-audit` — RISC↔RISC mailbox FIFO handshakes (push/pop balance, call-count symmetry, fence ordering caveat).
- **Cross-core (NoC):**
  - `dataflow-cb-sync-audit` — circular-buffer producer/consumer credits (reserve/push/wait/pop balance, data-before-credit ordering, capacity, remote CBs).
  - `noc-sync-audit` — raw `noc_semaphore_*` + barrier data-before-signal ordering and multicast fan-out, plus the read-side / exit / coherency half of the same surface (an inbound read consumed before its read-barrier, a non-posted atomic left in flight at kernel exit, and on Blackhole a hand-rolled L1 poll missing `invalidate_l1_cache`) — the non-CB half of dataflow.
- **Intra-thread (micro-architectural):**
  - `instruction-latency-audit` — pipeline result-latency / NOP padding on hand-written instruction sequences (compiler-grounded, arch-divergent).

## Recall preflight — run the `llk-audit` tool once, up front (augmentor, not a verdict)
Before fanning out, run the deterministic recall tool for all its checks in one
parse pass; it feeds every sub-audit a complete known-pattern worklist over one
shared fact base (which is also what makes the JOIN a lookup rather than a
schema reconciliation):

    tt_metal/tt-llk/.claude/tools/llk-audit/run.sh <wormhole|blackhole|quasar>
    # Run this from the TT-METAL REPO ROOT — the path above is relative to it, and it is
    # the usual session CWD. (In a session rooted at tt-llk itself, drop the
    # `tt_metal/tt-llk/` prefix; from anywhere else, prefix an absolute repo path.)
    # Invoke it BY PATH and do not `cd` first: loading a skill does NOT change the shell's
    # CWD, so a bare `cd .claude/...` fails from the repo root. No `cd` is needed anyway —
    # run.sh chdirs to its own directory, so where you invoke it from changes nothing.
    # CONSEQUENCE: every PATH you hand it (an out_dir, `LLK_KT_LOG`) must be ABSOLUTE —
    # a relative one resolves against the tool dir, NOT your CWD. Use `$PWD/<name>`.
    # PR-scoped sweep: add --changed [BASE] (default main) to scope every check to files changed vs BASE.
    # out/audit.<arch>.json -> .checks[{mmio-race, cfg-word-overlap,
    #                                    semaphore-handshake, reconfig-stall,
    #                                    srcreg-bank, mailbox-sync, cb-sync, noc-sync,
    #                              noc-atomic-exit, noc-read-barrier, noc-l1-invalidate}]
    #   (cb-sync + the four noc-* checks are committed + deterministic but need a
    #    KERNEL fact base to yield findings — over tt-llk all five are trivially
    #    empty; see kernel tier. mailbox-sync does yield its in-tree surface.)

Hand each sub-audit agent its check's `findings[]` as the pre-enumerated worklist,
and instruct it to **widen beyond the tool** per that check's `blind_spots` (the
tool recalls KNOWN patterns only — the agents must still hunt the unknown). The
tool ships committed deterministic checkers for **8 of the 9** classes — all but
`instruction-latency` (its surface is the SFPU files clang can't parse + its
verdict needs the out-of-tree pinned `sfpi-gcc` latency table → fully LLM-driven).
Of the 8, **`cb-sync` and the `noc-sync` class (4 checkers: `noc-sync`,
`noc-atomic-exit`, `noc-read-barrier`, `noc-l1-invalidate`)** only produce findings
when fed a **kernel fact base** (the committed kernel tier's capture — see
*Full-audit kernel tier* below); over the tt-llk fact base all five are trivially
empty, and without a capture run their kernel surface stays LLM-driven (each
skill's ttnn-widened grep).
`srcreg-bank` recalls only the dvalid control points + the raw-`SETDVALID`-on-BH
flag (not the bank-flip lockstep verdict), and `mailbox-sync` recalls only the
IN-TREE mailbox surface — mailbox use in ttnn/models kernels (one-to-one channels
and fan-outs) is covered by the skill's ttnn-widened grep unless a kernel capture
is run — so both still need heavy LLM widening per their `blind_spots`. The tool
is **advisory**: it never clears a class, and its silence is "no new
*known-pattern* instance," not "no bug".

**If `run.sh` exits non-zero, the sweep continues manually — a failed tool is not a
blocked audit** (the sub-skills already say "if unbuilt, proceed manually"; the
orchestrator must too). It refuses rather than emitting a false all-clear in four
distinct cases, and they mean different things: the **extractor won't build** (no
Clang/LLVM ≥ 18) or **no facts extracted** → every class is LLM-driven this run, say
so; a **`--changed` base that doesn't resolve** → fix the ref, never accept the empty
scope; a **kernel-tier failure** under `--full-jit` → the in-tree findings above it
still stand. In all four, report the tool status alongside the findings so no reader
mistakes "the tool didn't run" for "the tool found nothing".

**Coverage caveat — attend to `parse_errors` (a per-run partial-coverage signal).**
The envelope's `parse_errors` is `>0` on a normal run: the SFPU-heavy files that use
`sfpi::` types don't fully parse (expected — see `out/parse.log`), so writes in those
files are ABSENT from the fact base. That is a *known, expected* partial-coverage
bound, NOT a clean full-coverage guarantee — treat any class whose real surface is
SFPU-side as tool-under-covered there and widen manually. **But before trusting a
low/zero finding count, scan `out/parse.log` for a header that should NOT fail (a
NON-SFPU file):** an unexpected parse failure there is a silent coverage hole (that
class's writes in that file were never analyzed → its "0" is meaningless). This is
the consuming-side half of the never-false-all-clear contract.

**Read the envelope's `degraded` list too — it is the tool's own "I did not analyze
this" channel.** `cli.py` records there every surface it could not analyze this run
(empty/near-empty fact base, unreadable cfg defines, a `--changed` file that parsed
to 0 facts, a kernel-tier capture coverage hole, uncertain attribution under a
partial parse), and `run.sh` prints it as `*** DEGRADED — NOT a clean all-clear ***`.
A non-empty `degraded` makes a low/zero count **unanalyzed, not clean**: carry each
note into the report, treat the named surface as LLM-only for that class, and never
let a degraded run close a class.

**Tool-drift contract — detect a stale registry, surface it, offer the fix (do NOT silently edit).**
The deterministic tier is only as complete as `llkaudit/registry.py`'s name→meaning
tables. When the codebase adds a sync-relevant API the registry doesn't yet know —
a new `CircularBuffer`/`Noc` method, a `noc_async_*`/`noc_semaphore_*` variant, a
new cfg-write helper or `TTI_*` cfg instruction, a new mailbox call — the checkers
**under-recall silently**: fewer findings, no error, exit 0. A stale registry does
not look broken; it looks *clean*. So during any audit (especially a full sweep)
treat these as **tool-drift tells** and act on them:
- an `UNRESOLVED` bucket rising sharply in the checks that HAVE one — cfg-word
  (`UNRESOLVED`) and mailbox (`UNRESOLVED_ENDPOINT`) surface inputs they couldn't
  key; a jump means a new write/endpoint shape the registry doesn't classify;
- the kernel-tier **coverage ledger** showing TUs it could not translate/parse, or did
  not analyze cleanly for another reason — any row whose status LEADS with a hole marker
  (`PARSE-FAIL`, `PARSE-HOLE`, `EMPTY-OUT`, `EXEC-FAIL`, `SKIP-*`, `HOST-LEAK`,
  `NON-KERNEL-CMD`). Such rows are excluded from the ledger's own `ok/N TUs parsed`
  headline, and their COUNT is passed into the audit JSON's `degraded`. A *capture* hole
  (the checkers never saw those kernels), distinct from a classification gap but equally
  a silent miss;
- **the LLM tier finding a real sync site the deterministic worklist omitted** —
  the cleanest tell, and the ONLY one for **cb-sync / noc-sync**, which have no
  `UNRESOLVED` bucket (a CB/NoC object has many non-flow-control methods, so an
  unknown method can't be flagged deterministically without over-firing) — so a
  new `CircularBuffer`/`Noc`/`Semaphore` flow-control method shows up only as a
  site the LLM found that the tool's candidate list didn't.

**Before calling anything drift, check `KNOWN_GAPS.md`** (next to `registry.py` — the
tool's single canonical list of *deliberately deferred* gaps, each with its risk
class, live-today count, and why the obvious fix was refused). A gap listed there is
real but consciously not fixed, usually because the naive fix trades recall for
false-positives — so do **not** re-file it as newly-discovered drift: cite its `L#`/`X#`
and move on. If this sweep establishes a NEW gap the user chooses to defer, record it
**in `KNOWN_GAPS.md`**, not in the run report (a report rots; the ledger is read by
the next run).

On any such observation the ledger does not already own, **name the stale entry** (which `registry.py` table +
which checker under-recalls, and the concrete API missed), tell the user the
deterministic tier is drifting, and **offer to add it**. Apply the enhancement
**only if the user allows** — never silently. The fix is almost always a one-line
`registry.py` table entry (the single declarative edit point) plus a hermetic test
in `tests/test_checks.py`; rarely a checker change; **never** the C++ extractor
unless a genuinely new fact family is needed. Ground the new entry against the
authoritative header (e.g. `circular_buffer.h`, `noc.h`, `dataflow_api.h`,
`ckernel_ops.h`) per the source ladder, keep the ground-truth counts fresh, and
follow the commit-twice pre-commit discipline. Then re-run recall so the sweep
reflects the widened tables. This keeps the tool a living superset of the codebase
instead of decaying into a false all-clear as the APIs move.

**Drift runs both ways.** Under-recall (`CAP-REDUCTION`) is the more serious class, but
a checker can also **over-report** (`FALSE-FLAG`) — a flush or credit form its registry
doesn't yet recall, or a receiver-type text heuristic admitting a non-CB/non-NoC object.
So a tool `findings[]` entry is a **candidate, not a verdict**: confirm it at the site
before it enters a report. Dismissing an unconfirmed *tool candidate* with shown
evidence is **not** a monotonic-contract downgrade — that rule protects the sub-audits'
own verdicts (below), not the recall tool's pre-verdict worklist.

## Full-audit kernel tier (opt-in) — the committed JIT capture for cb-sync / noc-sync / noc-atomic-exit / noc-read-barrier / noc-l1-invalidate / mailbox-sync
The `cb-sync`, `noc-sync`, `noc-atomic-exit`, `noc-read-barrier`, `noc-l1-invalidate`, and `mailbox-sync` **checkers are committed and
deterministic** — but their kernel surface lives in **JIT-compiled kernels OUTSIDE
tt-llk** (`ttnn/`, `models/`, `tt_metal/hw/inc/api`), which have no static compile
database the in-tree fixed-flags parse can reach. So the checkers are always
present; what's missing at rest is a **kernel fact base** for them to run over.
The `kernel_tier/` module (committed in-tree) produces that fact base; the only
runtime-dependent step is the capture RUN itself.

> **What is durable vs runtime-dependent (the clean split):**
> - **DURABLE (committed, in the tool):** the `cb-sync` / `noc-sync` / `noc-atomic-exit` / `noc-read-barrier` / `noc-l1-invalidate` / `mailbox-sync`
>   checkers themselves — plain Python over a fact base, unit-tested, no JIT hook —
>   AND `kernel_tier/{capture.py,bootstrap.sh,MANIFEST}`, the capture pipeline.
>   Over the tt-llk fact base **cb-sync and all four noc-* checks are trivially
>   empty** (no cb/noc sites there) while **mailbox-sync yields its small in-tree
>   surface** (the MATH→UNPACK dst_index pair + debug endpoints); fed a kernel fact
>   base all six emit real candidates over the outside-tt-llk kernels.
> - **RUNTIME-DEPENDENT (each sweep):** the capture RUN — producing a build log
>   that carries the JIT compile commands, then translating each RISC-V-GCC command
>   to clang. This needs a build log or a live runtime, and the GCC→clang
>   translation is the fragile part — isolated in `capture.py` with an honest
>   coverage ledger (untranslatable TUs are listed, never silently dropped). No
>   `ccwrap` / compiler-wrapper and no `jit_build` patch: capture is just a log scrape.

**CURRENT STATE:** the module is **committed and in-tree**. `run.sh
--kernel-tier-status` prints **`available`**, and `run.sh --full-jit` runs the
in-tree audit **and then** `kernel_tier/bootstrap.sh`. It never captures silently:
bootstrap needs either a pre-captured log (`LLK_KT_LOG`) or permission to run a
workload (`LLK_KT_WORKLOAD` [+ `LLK_KT_CLEAR_CACHE=1`], which needs a device/sim).

**When to offer it — gate on BOTH conditions:**
1. **Mode:** the user asked for a **full / exhaustive** sweep (NOT a diff/PR-scoped
   `--changed` run). A diff/PR run **never** prompts — the fast path stays clean.
2. **Runtime:** the capture needs a build log or a device/sim. If neither is
   available, do **NOT** silently improvise. Proceed with the in-tree sweep and
   **tell the user precisely what that means for cb-sync / noc-sync / noc-atomic-exit / noc-read-barrier / noc-l1-invalidate / mailbox: they
   are NOT tool-recalled *over kernels* this run, but they are STILL AUDITED —
   LLM-driven via each skill's (ttnn-widened) grep + reasoning + ISA docs. "Not
   tool-recalled" ≠ "left out."** The only thing forgone is the extra deterministic
   candidate list (grep-fidelity recall, so macro-wrapped/aliased calls and
   cross-kernel pairing may be missed — state that as the coverage bound).

**Cost honesty before running it:** capturing by running a workload needs a
**runtime** (hw/sim), runs a workload build (minutes, and `LLK_KT_CLEAR_CACHE=1`
forces op-kernel recompilation), and gives **periodic-sweep-grade** coverage that
is **complete only over the kernel variants actually exercised** (a clean result
must never read as "all kernels covered").

**How to run the capture** (the module is already committed — you only RUN it):
1. **Get a build log** carrying the JIT compile commands — either
   `TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 <workload> > $PWD/build.log 2>&1` captured on
   hardware once (then audit offline), or let bootstrap run the workload for you.
2. **Run the tier** — same `run.sh` path as the preflight above, from the repo root, and
   an **absolute** `LLK_KT_LOG` (per the preflight's path rule; a relative one resolves
   against the tool dir and fails as "not found"):
   `LLK_KT_LOG=$PWD/build.log tt_metal/tt-llk/.claude/tools/llk-audit/run.sh <arch> --full-jit`
   (log path), or the run-a-workload path:
   `LLK_KT_CLEAR_CACHE=1 LLK_KT_WORKLOAD='<cmd>' tt_metal/tt-llk/.claude/tools/llk-audit/run.sh <arch> --full-jit`.
   bootstrap → `capture.py` (scrape → GCC→clang translate →
   `llk_extract` per kernel → merged fact base) → cb/noc/read/atomic/l1/mailbox over it.
3. **Coverage is emitted automatically:** `kernel_coverage.<arch>.txt` lists every TU
   as parsed or with a leading hole marker (`PARSE-FAIL`, `PARSE-HOLE`, `EMPTY-OUT`,
   `EXEC-FAIL`, `SKIP-*`, `HOST-LEAK`, `NON-KERNEL-CMD`) — no silent caps. Holes reach
   the audit JSON's `degraded`, so *some* kernels failing can't be hidden by others
   succeeding; an empty fact base makes bootstrap exit non-zero outright rather than
   emit a false all-clear. **Read `degraded` before treating 0 findings as clean.**
4. **Merge:** merge/dedup the kernel-tier candidates with the in-tree findings.

Everything lives **inside tt-llk** (`.claude/tools/llk-audit/`), so there is zero
permanent footprint outside tt-llk. Whatever the kernel tier surfaces is
**candidates** (augmentor) — the data-before-credit ordering, cross-kernel
producer↔consumer pairing, and mailbox call-count-symmetry/ordering **verdicts
stay with the sub-audit skills**.

## The monotonic contract (non-negotiable — this is what makes the sweep a true superset)
A naive "run them + concatenate" can catch *less* than the audits alone (summarization loss, dedup collapse, over-resolution). To prevent that, the JOIN is **additive-only**:
1. **Preserve every per-audit finding verbatim.** The output *includes* all nine raw reports (full finding lists, not summaries). Nothing is dropped, merged-away, or reworded.
2. **The JOIN may only ADD findings or ESCALATE severity.** It must never silently delete or downgrade a per-audit verdict.
3. **No silent downgrades.** If the JOIN verifies a cross-reference and believes a flagged item is actually safe, it attaches an *annotation* next to the original flag with the shown evidence — it does **not** replace the flag. Default to keeping the flag; never upgrade "probably safe" to SAFE without proof at that exact site.
4. **No summarization at the fan-out boundary.** Sub-audits return their full enumerations + verdicts; the JOIN reasons over those, not over compressed digests. If you fan out to agents, instruct each to return the complete finding list (and its candidate count) so the JOIN can re-judge dismissed sites if a cross-reference implicates them.
5. **No cap without a closer.** If coverage is bounded anywhere (top-N, sampling, agent budget, a file left unopened, a verdict resting on an unconfirmed claim), `log`/state it **AND treat it as a work item, not a terminal state**: an exhaustive run must spawn a gap-closer for each bound and **loop closers until the bound is gone** (see *Exhaustive coverage — file manifest + ledger* below). A bounded sweep must never read as exhaustive, and "I sampled N of M / deferred the experimental files" is an **incompleteness that blocks the done-verdict**, not an acceptable caveat.

## Method
1. **Run all nine audits** (faithful execution — invoke each skill or run its deterministic enumeration; don't approximate). Collect each one's complete finding list with per-finding `file:line`, verdict, and — critically — its **stated assumption** for every SAFE/LATENT verdict (the "safe because …" clause).
2. **Build the cross-reference worklist.** Extract every verdict whose safety is conditional on another class's invariant. The known seams (starting set — not exhaustive):

   | Audit says… | …safe because (other-class invariant) | JOIN must verify |
   |---|---|---|
   | `cfg-word-overlap`: shared SrcA/SrcB ALU-format word is LATENT | "pipeline semaphores keep the reconfig from overlapping the other thread's op" | the *specific* format RMW sits inside the region that `semaphore-handshake` proved ordered — same semaphore, write between wait and post/get, on every branch |
   | `mmio-race`: MMIO config write SAFE | "a semaphore / STALLWAIT(TRISC_CFG) orders it before the consumer" | `semaphore-handshake` shows that semaphore is balanced+init'd AND the MMIO store is sequenced relative to the post; or the stall's condition actually covers the consumer |
   | `reconfig-stall`: per-thread drain present (e.g. `STALLWAIT(STALL_CFG, PACK)`) | (drains *this* thread's unit only) | does another **thread** write the same word? → hand to `cfg-word-overlap`; a per-thread drain never excludes a cross-thread writer |
   | `semaphore-handshake`: semaphore protocol SAFE | (verifies counting, not payload) | which config words/dest/src rely on this semaphore for mutual exclusion? → confirm each such write is actually inside the ordered window |
   | `mailbox-sync`: mailbox handshake SAFE | "the memory the mailbox value refers to is ready, and all threads reach the mailbox handshake equally (including hand-written mailbox_write in ttnn/models kernels)" | the referenced memory (L1 tile, dest offset) is ordered-ready — a plain `fence` does NOT order a mailbox write against a prior store to a different region (a no-op on WH; on BH it drains the store queue but not to *processed*), so cross with `mmio-race`/memory-ordering AND hand the "is the CB page ready?" half to `dataflow-cb-sync`; and the call-count symmetry holds on every branch (same control-flow that `semaphore-handshake` balance depends on) |
   | `dataflow-cb-sync`: CB credit SAFE | "the page write is ordered before the credit, and reserve/wait gates the access" | the data-before-credit barrier (NOC flush before `cb_push_back`) is present → cross with `mmio-race`/NOC ordering — this holds for the Metal 2.0 `DataflowBuffer` object form too, whose credit quartet IS recalled but whose per-buffer `dfb.write_barrier(noc)` is NOT recognized as a flush, so confirm the flush at the site instead of trusting a no-flush candidate; the address `mailbox-sync` sends (over its directed tile-address channel) derives from `fifo_rd_ptr` gated by this `cb_wait_front`; and `tile_regs_*` interleaving is `semaphore-handshake`'s `MATH_PACK` |
   | `mmio-race`: MMIO-vs-MOP write SAFE | "a `mop_sync()`/`tensix_sync()` drains it" | the drain provably covers the consumer at the site (right primitive, every path, cross-call window). ALSO: is the drain heavier than needed (OVER-SYNC) or unnecessary (REDUNDANT)? → perf finding, never suppresses the race verdict |
   | `srcreg-bank-sync`: SrcA/SrcB bank handoff SAFE | "the FPU op waits for `AllowedClient`, and Dst/LReg is ordered" | bank-flip is lockstep on both sides; the Dst/LReg half rides `MATH_PACK`/`mutex::SFPU` → hand that half to `semaphore-handshake`; single-thread ownership of the bank state holds |
   | `noc-sync`: cross-core credit SAFE | "the remote write is flushed before the credit, and the wait count matches the fan-out" | a WRITE credit (`set_remote`/`set_multicast`/`relay_*` — **bare `noc_semaphore_set` is a LOCAL reset store, not a credit**) same-NoC/VC/dest is ordered by issue-order **only when payload and credit target the same memory kind**: per `Ordering.md`, an L1 write followed by a write to an **MMIO** address (a stream/config register — the `noc_inline_dw_write` shape) has **no** ordering guarantee and the MMIO write can race ahead, so a register-target credit needs a barrier like an atomic one; an ATOMIC credit (`noc_semaphore_inc`/`inc_multicast`/remote `up`) needs the payload write **committed** — `noc_async_write_barrier` (ACK), not a bare `writes_flushed` (departure only, per `data_movement_doc/general/posted_writes.md`). Do NOT clear a flush-only atomic (the checker tags it `FLUSH_NOT_BARRIER`, or `POSTED_FLUSH_ONLY` when the only preceding flush drains the posted-write counter — a no-op for a non-posted write/inc) as safe on a same-VC-unicast assumption — confirm against `<arch>/NoC/Ordering.md` + `posted_writes.md`. Cross with `dataflow-cb-sync` when the same buffer is also a CB page |
   | `dataflow-cb-sync`: credit-counter read SAFE | "the counter read observes the other side's update" | on **Blackhole** that is a *cache*-coherency claim, not a CB-API one: a HAND-ROLLED poll of a CB/credit word (anything not going through `cb_wait_front`/`cb_reserve_back`) needs `invalidate_l1_cache` — the invariant `noc-sync`'s L1-coherency lens owns → hand it there. Latent while the RISC L1 D-cache is off by default; a real hang once it is enabled |
   | `instruction-latency`: sequence SAFE | "the compiler scheduled the NOPs" / "Blackhole HW scoreboards it" | the code is actually sfpi-compiled (provenance lens), not raw `TTI_*`; and for BH the consuming insn is NOT in the freshly-derived `xtt_dynamic_bug` errata set — re-derive from the pinned `sfpi-gcc`, never a baked list |
   | any: "value-invariant / unit-idle / single-thread" | (assumption about another class's state) | re-confirm the assumption at the site with the other audit's lens |

3. **Discharge each cross-reference at its actual site** (read the code; don't trust the summary). Trace the one physical resource (a CFG word, dest/src bank, a semaphore) across *all* mechanisms that are supposed to guard it. A race exists when the guarantees, composed, leave a gap — even if each guarantee is individually valid.
4. **Emit** per the monotonic contract: nine raw reports untouched, plus a new **EMERGENT** findings section, plus any escalations (annotated).

## Verdict (combined)
- **EMERGENT-RACE** — individually all-SAFE, but the join condition is unmet at a reachable site (e.g. format RMW moved outside the semaphore window; MMIO write not actually sequenced to its gating post). Real; report with the full cross-class chain.
- **Per-class verdicts** — passed through unchanged from each sub-audit (SAFE / RACE / LATENT / HARDENING-GAP / INIT-BUG …).
- **ANNOTATED-SAFE** — a per-class flag the JOIN cross-checked and believes is discharged; keep the original flag, attach evidence, mark for maintainer confirmation. Never silently resolve.
- **UNCERTAIN — needs HW/owner confirmation** — the label mandated by *Never infer a NEGATIVE from a missing doc* (below) when no reachable authority resolves the point. It is a **surfaced, non-closed** finding: it counts in the totals, is neither RACE nor SAFE, and must never be silently promoted to either. A JOIN seam whose invariant could not be discharged lands here, not in EMERGENT-RACE.
- **FALSE-POSITIVE (unreachable)** — a value-gated hazard ruled out by a host-side invariant (program factory / device-op validation / `constexpr`), per *Confirm reachability* below. Report it with the invariant that rules it out, so the next run doesn't re-raise it.

## Ground-truth freshness contract (non-negotiable)
Two sub-audits depend on sources that are **dynamic** and must be consulted live each run, never from data baked into a SKILL.md:
- `instruction-latency-audit` grounds in the **`sfpi-gcc` version this build is pinned to** (latency tables, `xtt_delay`, `xtt_dynamic_bug` errata) — re-derive every run; optionally diff the tip and flag divergence; mark coverage bounded if the pinned compiler can't be resolved.
- All HW-semantics verdicts ground in the **tt-isa-docs MCP** (fetched live), not cached prose.
Any instruction list / latency number / errata set appearing in a sub-skill is a *dated illustration*, subordinate to fresh derivation. The JOIN must not treat a stale baked list as authority.

**ISA-precedence policy (applies to every sub-audit):** the live ISA doc (and, for `instruction-latency`, the pinned `sfpi-gcc`) **outranks** any rule, table, or example baked into a skill. When a live source **contradicts** a baked rule, do NOT silently resolve it — surface the conflict to the user and ask whether the baked rule should be overwritten, discarded, or kept; default to the live source. This holds whether a sub-audit is run standalone or inside this sweep.

**Ground-truth source ladder (a SUPERSET of the sage agents' corpus; `assembly.yaml` deliberately excluded) — use every applicable source each run and combine them:**
Authority is **per audit-class × arch**; never skip a source because it lacked coverage last time. HW facts are confirmed against an authoritative source, never extrapolated.
**In-repo code is a first-class living source, but only for facts that ARE the code** (applies to every arch below): instruction existence/encoding — `tt_metal/tt-llk/tt_llk_<arch>/common/inc/ckernel_ops.h` (a `TT_OP_*`/`TTI_*` with an opcode = valid on that arch, regardless of ISA-doc coverage), and *what a call actually issues* — VC / cmd-buf / posted / barrier — `tt_metal/hw/inc/api/dataflow/dataflow_api.h` + the per-arch `tt_metal/hw/inc/internal/{tt-1xx/<arch>,tt-2xx/quasar}/noc_nonblocking_api.h`. Code is **NOT** authoritative for HW *semantics* — whether HW then orders/latches those transactions, latency, sampled registers, errata — that is the ISA doc's domain, and existence-in-a-header is never a reason to skip it; **nor** for the correctness of the audited kernel (self-certification is circular). A code **comment** is only an author's belief, never ground truth (e.g. an in-repo `#ifdef ARCH_BLACKHOLE` comment asserting BH reorders across cmd buffers was wrong — BH is `noc2axi`-ordered; ground ordering in `<arch>/NoC/Ordering.md`, not the comment).
- **HW-semantics audits** (mmio, reconfig, cfg-word, semaphore, mailbox, srcreg, noc, dataflow-cb):
  - **WH B0 / BH A0:** **tt-isa-docs MCP** *or* **DeepWiki** — both serve the same `tenstorrent/tt-isa-documentation` corpus, either is fine — as primary ISA; **code** (per the code principle above) to cross-verify. **Fetch ISA pages by EXACT path (raw GitHub, or `gh api repos/tenstorrent/tt-isa-documentation/contents/<path>`), NOT index search** — *both* index-backed routes (semantic doc search AND repo code search) silently miss real pages, and can return fully empty when the server's upstream read of the corpus fails, with no error to distinguish that from a genuine miss (a prior audit wrongly concluded `<arch>/NoC/Ordering.md` "did not exist" and inferred a race from the gap). A miss from either is **NO EVIDENCE**, never absence; canary-gate them per the *Source preflight* below. **WH/BH Confluence:** one canonical pointer — `1001357404` → **noc / dataflow-cb** HW-bug workarounds (WH/BH); open at audit (opaque ID, don't transcribe).
  - **Quasar:** **tt-isa-docs MCP** — *queried every run* (empty for QSR today, but coverage is expected; promote to primary the moment it lands; do **not** use DeepWiki for QSR — it has none); **Quasar Confluence HW pages** (internal/authenticated only) as the de-facto primary today — **don't rely on naïve CQL search; anchor to these canonical pages** (seed list, not exhaustive; run a freshness check before citing any — pages older than ~3 months get a staleness caveat, >9 months or undated must be re-verified against code/HW).
    These are **pointers** — open the page in Confluence at audit for its contents; do not transcribe page titles or page facts into reports or skills. The page IDs (kept opaque on purpose — open them at audit):
    - `1340276980` → grounds **cfg-word / mmio / reconfig**
    - `1613201604` → per-instruction semantics for any class
    - `586515553` → Quasar↔WH/BH arch deltas
    - `646217858` → **semaphore / noc**
    - `113017320` → **srcreg / semaphore**
    - `1256423592` → **instruction-latency** HW context
    - `84508873` → general arch overview
    - `627475207` → **noc / dataflow-cb** HW-bug + feature tracker (Quasar; historical/bring-up — verify against code at audit)

    Search beyond these when a topic isn't covered, but start here. If a seeded ID returns 404 (page deleted/recreated), re-find the page in Confluence by topic and update the ID. **BH-inference** only as a caveated last resort (never grounds or overturns a verdict). Where two sources carry a fact, cross-check and surface conflicts.
- **instruction-latency (all archs):** the pinned **`sfpi-gcc` source** (`rvtt.md` / `sfpu-ops-{wh,bh,qsr}.h` / `rtl-rvtt-schedule.cc`) — **fetch it at the pin**; the compiled toolchain / a compile experiment is not a substitute. Secondary: tt-isa-docs `VectorUnit` (WH/BH HW latency) and the Quasar SFPU uArch Confluence page (QSR HW context). (Instruction existence/validity → `ckernel_ops.h` per the code principle above, never ISA-doc coverage.)
- **dataflow-cb / noc:** the **dataflow API source** (per the code principle above — authoritative for *what a call issues*: VC / cmd-buf / posted / barrier) plus the per-arch ISA sources above. **NoC transaction ordering** grounds on the ISA `<arch>/NoC/Ordering.md` page (fetch by exact path per the WH/BH note; the WH page carries the same-source/dest/NoC/VC in-order rule, write→atomic having *no* ordering rule, and response/ack always reorderable). **Blackhole (re-check every run — a 404 is point-in-time, not permanent):** fetch `BlackholeA0/NoC/Ordering.md` each run; the moment it exists it becomes the authority and supersedes the fallback below. While it is absent (404 at time of writing), ground BH ordering on code + `1001357404` + the `noc2axi` ordering fact confirmed by the SOC/NoC team, marked a coverage caveat. Secondary OP-writer-level tier: the `tenstorrent/tt-low-level-documentation` repo `data_movement_doc/` (barrier taxonomy, posted-vs-nonposted, VC-vs-credit tension). Quasar adds the tile-counter / global-semaphore Confluence pages.

**Ground-or-abstain (this is what makes a re-run a superset of any prior run):** ground each verdict against the *applicable* authorities at full strength; if **none** is reachable, **emit no verdict** for that point (label the coverage hole, e.g. `[Quasar: code-only]`) — never substitute a weaker basis (WH/BH extrapolation, a code comment, a compile experiment) for a verdict, and never let a weaker-grounded result overturn a stronger-grounded one.

**Never infer a NEGATIVE from a missing doc (guards against the dominant false-positive class).** A missing ISA-doc page means *undocumented*, never *absent/invalid/unordered* — the corpus is an explicitly incomplete living document. So doc-silence must NOT become a confirmed `RACE`/`INVALID` verdict, and equally must NOT become `SAFE`. Before asserting any negative ("instruction invalid on arch Y", "ordering not guaranteed → race"): (1) first try to *resolve* it against the applicable authority in the ladder above (e.g. instruction existence → `ckernel_ops.h`; HW ordering → `<arch>/NoC/Ordering.md`) and take the resolved verdict if found; (2) if no reachable source resolves it, emit **UNCERTAIN — needs HW/owner confirmation** (a surfaced, non-closed finding), *not* a confirmed race and *not* safe. This is not a relaxation of conservatism: the item is still raised for confirmation (nothing escapes), it is only labeled by true confidence. A fabricated mechanism filed as "confirmed" (e.g. a "BH reorders cross-cmd-buffer → RACE" claim later refuted by the SOC team as `noc2axi`-ordered) is *anti*-conservative — each refuted confirmed-flag trains owners to dismiss the auditor, so real races later get waved off. **Record** which source + revision (ISA-doc rev / DeepWiki, Confluence page + date + freshness, sfpi-gcc commit) each verdict was grounded against, so runs are reproducible and comparable. (A stateless run cannot *recall* a prior one — to be a superset of a **specific** prior run, supply that run's report as an input and reconcile against it monotonically.)

**Confirm reachability before flagging a value-gated race/deadlock (a positive verdict needs grounding just as much as a negative).** When a hazard depends on a specific runtime value — a loop count of 0, a dimension of 0, an empty-range iteration, an `== N` branch — the code path being statically present is NOT sufficient. Verify the triggering value is actually reachable given the op's **host-side invariants**: the **program factory** (compile-arg derivation, grid/shape math), the **device-op validation** (`TT_FATAL`, layout requirements — e.g. a TILE-layout requirement forces tile-aligned dims ≥ 32), and compile-time constants (`constexpr`). If a repo invariant makes the triggering value impossible, the finding is a **FALSE-POSITIVE (unreachable)** (the verdict label above) — or at most a latent "if this invariant is ever relaxed" note — never a confirmed race. A kernel cannot be audited in isolation from the factory that launches it; this is the same ground-in-the-authoritative-source discipline, with the factory + device-op validation as the source. Examples: a `Ht==0` deadlock ruled unreachable because the op `TT_FATAL`s TILE layout (so `Ht ≥ 1`); an off-by-`start_core_id` loop bound that is correct because `start_core_id` is `constexpr 0`.

**Source preflight — emit this FIRST, before any audit work, and PAUSE for the user.**
Before running any sub-audit, build a **source manifest** from the ladder above and **probe each source's reachability**, then present it and let the user decide. Render a table — *source · tier/role · arch+class served · reachable? (✓ / ✗ / N-A) · note* — covering at least:
- **tt-isa-docs MCP** (WH/BH ISA; also queried for Quasar — expected empty today) — callable? **Probe with a CANARY, never a bare call.** It is a third-party proxy over the ISA corpus, and when its upstream read of that corpus fails it answers a *successful empty result* — "No code matches found" / "No documentation found", `fileUsed: "unknown"` — not an error, which is indistinguishable from a real miss and reads as "the page does not exist". So query the index for a term that CANNOT be absent (e.g. `Tensix`): empty canary ⇒ the **index is down**, mark the row `✗ [index down]`, and for the rest of that run treat every index miss as **NO EVIDENCE** (never absence — see *Never infer a NEGATIVE from a missing doc*). Report the two routes as SEPARATE rows, because they fail independently: **index search** (canary-gated) and **exact-path fetch** (raw GitHub / `gh api …/contents/<path>`), which bypasses the proxy's corpus index and so survives its outages. Exact-path reachable ⇒ no verdict is bounded by a dead index. Both routes dead ⇒ WH/BH lose their primary ISA authority, so abstain per *Ground-or-abstain* — except where the ladder names another WH/BH source (the `1001357404` Confluence pointer for **noc / dataflow-cb**), which can narrow such a verdict but never replace the ISA page.
- **DeepWiki MCP** (WH/BH ISA, equivalent corpus) — callable? Same proxy class, so **same canary rule + same `[index down]` label**; being the redundant route for the same corpus, it is only a coverage loss if tt-isa-docs' exact-path fetch is ALSO unreachable.
- **Confluence / Atlassian MCP** (Quasar HW; internal-auth) — authenticated & reachable?
- **sfpi-gcc source** (instruction-latency) — resolve the pin: `tt_metal/sfpi-version` gives an `sfpi_version` tag → that tag of `tenstorrent/sfpi` → its `gcc` submodule commit in `tenstorrent/sfpi-gcc`; confirm the files fetch (the exact three-step recipe lives in `instruction-latency`'s freshness contract — follow it there, don't re-derive).
- **dataflow API source** (`tt_metal/hw/inc/api/dataflow/`) — present?
- **code** (`tt_llk_*`) — present (≈ always).

Then state, **per unreachable source, which verdicts will be bounded or abstained** (e.g. Confluence unreachable → Quasar HW = `[code-only]`; sfpi-gcc unfetchable → latency abstains). **Pause and ask the user to choose:** (a) **proceed** accepting the listed coverage bounds, (b) **help reach a missing source** (authenticate Confluence, grant network, provide a local path/clone), or (c) **add a new source** to the ladder. Do not begin auditing until the user chooses; re-list if the reachable set changes mid-run.

**Where it sits in the run order:** source preflight (this table) → **pause for the user's choice** → *Recall preflight* (the `run.sh` invocation above; it needs no source, so it may also run first to fill the table's "code" row) → fan-out → JOIN. The pause gates the *auditing*, not the tool run.

**Runs once per sweep.** In a fanned-out `race-audit-all` run the **orchestrator** performs this preflight a single time *before* fan-out and passes the confirmed sources + the user's choice into each spawned sub-audit's prompt; the spawned sub-agents **must NOT re-run the preflight or pause** — they audit against the already-confirmed sources. A sub-skill runs its own preflight only when invoked **standalone**.

## Coverage & execution

**Coverage — floor, not ceiling (applies to every sub-audit and the JOIN):** the grep patterns, site lists, and seam tables in this suite are a **seed, not an exhaustive enumeration**. Treat them as a minimum: after running them, widen with full reasoning. The techniques named here are **illustrative, not the allowed set** — use any approach your reasoning suggests, including ones not listed: e.g. semantic search (by behavior/effect, not token), resolving macros/wrappers/typedefs/indirection the literal patterns miss, following the call graph across files/layers, and diffing WH/BH/QSR variants. Pursue and report any hazard, primitive, seam, or site the encoded method doesn't cover — by any means; a more capable analysis must **not** be clamped to what is written here or to these techniques. The encoded patterns lower-bound coverage and reduce variance — they do not cap it. State residual coverage gaps explicitly (no silent caps).

**Exhaustive coverage — file manifest + ledger (MANDATORY when an exhaustive / "no-skip" run is requested).** Before fan-out, enumerate the **complete file list in the declared scope** — glob every `*.h`/`*.cpp` under each in-scope tree (`tt_llk_{wormhole_b0,blackhole,quasar}`; for the dataflow classes also `tt_metal/hw/inc/api/dataflow/` + the enumerated ttnn kernel families) — and emit it as a **coverage ledger**. Every file must finish the run in exactly one state: `audited` (a cell actually read it), `abstained` (named reason — e.g. no reachable authority for its arch), or `out-of-scope` (named reason). **The run may NOT report "done" while any in-scope file is `not-opened`.** This is what converts "sampled 6 of 186 / experimental not opened" from an acceptable footnote into a blocking incompleteness. Exhaustiveness is always *relative to the declared scope*: the ledger makes the scope boundary **explicit** (shown, not silently drawn by sampling) and proves nothing inside it was skipped. Report the scope boundary and the per-state ledger tallies in the Output.

**Execution — parallel by default.** Default to **concurrent `Agent` fan-out**, scaled to the candidate set and saturating the ~10–16 concurrency cap; inline only for a trivial diff. **The fan-out unit is `class × arch × file-or-subsystem`, not `class`** — one agent per (class, arch, file-group), sized to the manifest above. For the exhaustive tier a single agent **must not cover more than one architecture**: covering WH+BH+QSR in one context dilutes the weakest-grounded arch and is the prime cause of skipped Quasar coverage. Concurrent `Agent` fan-out needs no opt-in; the heavyweight **Workflow** tool remains the explicit-opt-in exhaustive tier. The JOIN/synthesis stays sequential. Don't over-spawn small work — but **never collapse the per-(class,arch,file) fan-out into a handful of inline all-arch agents to save the concurrency cap**; that trade is exactly what makes files sampled-and-caveated instead of opened. **If the session forbids subagents** (a harness/user policy against spawning), run the cells inline **serially at the same per-(class,arch,file) depth** and say that the run is slower for it — the coverage rules (ledger, no-cap-without-a-closer, one arch per context) still bind; losing parallelism must not become losing coverage.

**Persisting results — single writer, incremental.** Agents only **return** their findings; they never write a shared file (no concurrent-write clobbering). If findings are persisted to a file, the orchestrator/caller is the **sole writer** and **appends each wave's returns as they arrive** — incremental, never only-at-the-end — so an interrupt preserves every completed wave's findings (you lose at most the in-flight wave).

## Architecture note
WH/BH: all nine classes apply; cross-references as above. **Quasar**: HW AutoTTSync changes the RISC↔Tensix MMIO-ordering class (WH/BH need manual ordering; read Confluence `1340276980` at audit for what AutoTTSync actually guarantees), so seams touching `mmio-race` resolve differently — and note the recall tool blanket-tags QSR cfg/GPR writes `AUTOTTSYNC_ORDERED`, i.e. **pre-clears them out of `findings[]`**, while TTSync's RQ tracking *excepts* `MOP_CFG` / `REPLAY(load=1)` / `RESOURCEDECL` / post-load-replay consumers: a QSR mmio seam must be discharged against the actual consumer, never against the tool's silence; `instruction-latency` is also arch-divergent (BH/QSR scoreboarding vs WH always-pad). The cfg-word / semaphore / reconfig / mailbox / dataflow-cb / srcreg-bank / noc seams still apply (verify Quasar mailbox + NoC + unpack-to-dest HW semantics before extending verdicts there; the CB API is arch-agnostic but its NOC ordering primitives are arch-specific). Each sub-audit carries its own Quasar caveat — honor them in the join, and ground every HW claim per the **Ground-truth source ladder** above (a superset of the sage agents' corpus, `assembly.yaml` excluded — tt-isa-docs/DeepWiki for WH/BH, tt-isa-docs+Confluence for Quasar, sfpi-gcc for latency, BH-inference caveated-last-resort), ground-or-abstain — flagging any fallback verdict as such.

## Thoroughness — exhaustive tier (Workflow pipeline)
**Default = parallel**, per the *Execution* rule above; the JOIN then runs inline, since it must follow the per-audit results.

**Exhaustive tier (opt-in: the user asks for an exhaustive / "no-skip" run, or for multi-agent orchestration).** Run a **Workflow** with the pipeline below — it is the only mode that actually satisfies the *file-manifest + ledger* and *no-cap-without-a-closer* rules; a handful of inline all-arch agents cannot, and that is the failure mode this tier exists to prevent:
- `phase 0 — Enumerate`: build the **file manifest / ledger** (above) + derive the pinned `sfpi-gcc` tables once + enumerate the dataflow/ttnn kernel families.
- `phase 1 — Deep audit`: one agent per `(class, arch, file-group)` cell — HW-semantics cells use the matching **`sage-{arch}`** agent (so the weakest-grounded arch gets equal depth), instruction-latency/dataflow cells use a Bash-capable agent (for `gh`/grep). Each returns its FULL finding list + candidate count + its own stated coverage gaps, schema-structured. Mark every file it read `audited` in the ledger.
- `phase 2 — Verify (MANDATORY, adversarial)`: one verifier per finding that is non-SAFE **or** whose `safe_because` cites a doc claim, an arch analogy, or "HW not confirmed". The verifier tries to REFUTE; for a doc-claim SAFE it checks the doc's *mechanism actually maps to the instruction emitted* (e.g. `cfg_rmw`=RMWCIB is a Tensix instruction, NOT MMIO → shadowing/TTSync do not order it). Default to keep-flag when it cannot prove safety — and split by *why* it failed: the site showing the join condition unmet is **EMERGENT-RACE**, whereas no reachable authority to decide is **UNCERTAIN** (never silently the other).
- `phase 3 — Reconcile + critic`: if a prior run's report is supplied, reconcile **monotonically** (every prior finding → confirmed / refuted / false-positive-correct / **not-found**). A completeness critic per class reads the ledger + stated gaps and emits concrete closer work-items.
- `phase 4 — Closer loop (until ledger clear)`: spawn a closer per `not-found` / open gap / `not-opened` file; **repeat until no in-scope file is `not-opened` and no gap is open** (loop-until-dry) — or until a user-set budget is hit, in which case state the residual count explicitly (then it is a *bounded* run, not an exhaustive one).
- `phase 5 — Dedicated thin-grounding pass`: re-run, at the same per-file depth, the architecture with the weakest live grounding (today **Quasar**, Confluence-only) — the gaps a single all-arch pass leaves are concentrated there.
- `phase 6 — Synthesize`: union the raw findings; the JOIN only **adds/escalates** (monotonic). The cross-reference adversarial check (`phase 2`'s join-condition refutation) and the completeness critic ("which 'safe because' clauses were never discharged? which resource wasn't traced across all mechanisms?") are part of phases 2–3.

## Output
Where it goes: the **raw reports** are bulky (a fanned-out run returns many per class × arch × file-group), so persist them to the run's report file — the sole-writer file of *Persisting results* above, unioned per class, still verbatim per finding — and put the **synthesis** (items 2–6) in the reply, with the file path. (A small inline run may keep everything in the reply.) "No summarization" governs the *record*; it does not require pasting every enumeration into the chat.
1. **Nine raw reports**, verbatim, one section each (union the per-cell returns under their class; never compress a finding).
2. **Cross-reference worklist** — every "safe because <other class>" clause and whether it was discharged.
3. **EMERGENT-RACE findings** — cross-class chain (`file:line` → resource → the composed guarantees → the gap) + fix.
4. **Escalations / ANNOTATED-SAFE** — additive only.
5. **Totals** per verdict per class + emergent count — **including the `UNCERTAIN` and `FALSE-POSITIVE (unreachable)` counts** (an audit that reports only RACE/SAFE is hiding its own confidence bounds) — and an explicit note of any coverage bound. State plainly that no per-class finding was dropped or downgraded.
6. **Coverage ledger** (exhaustive runs) — the declared scope boundary + per-state file tallies (`audited` / `abstained` / `out-of-scope`), and **0** `not-opened` (or, if a budget was hit, the explicit residual count — marking the run *bounded*, not exhaustive). If a prior run's report was supplied: the reconciliation table (every prior finding → confirmed / refuted / false-positive-correct / not-found).
