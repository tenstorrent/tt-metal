---
name: srcreg-bank-sync-audit
description: Audit the shared backend DATA registers — SrcA/SrcB bank-valid (AllowedClient) + bank-flip handshake between unpacker and Matrix Unit, and the shared-once Dst/LReg overwrite hazards not already carried by the MATH_PACK semaphore or mutex::SFPU. Use after touching unpack→math dataflow, SETDVALID/CLEARDVALID, bank-flip bookkeeping, MOVD2A/MOVA2D/MOVB2D, or any cross-thread Dst/LReg access.
user_invocable: true
---

# /srcreg-bank-sync-audit — shared backend data-register hazards

> **Ground-truth precedence:** the live ISA doc (tt-isa-docs MCP, fetched each run) outranks every rule, table, and example baked into this skill — treat those as dated illustrations. If the live ISA doc **contradicts** a baked rule here, do NOT silently proceed: surface the conflict to the user and ask whether the baked rule should be overwritten, discarded, or kept. Default to the ISA doc.
>
> **MANDATORY — before any verdict, read the shared grounding policy.** The per-architecture **source ladder** (which docs to consult), the **ground-or-abstain** rule, and the **Source preflight** (list the sources you'll consult with their reachability + hierarchy, then PAUSE for the user) are defined once in `race-audit-all` → `.claude/skills/race-audit-all/SKILL.md`. **Your FIRST action is to `Read` that file and follow its "Ground-truth source ladder", "Ground-or-abstain", and "Source preflight" sections** — they are load-bearing: a verdict produced without them is ungrounded and MUST NOT be reported. If that file genuinely cannot be read, say so and **abstain** rather than proceed ungrounded. (If you were spawned by a `race-audit-all` sweep — your prompt already lists the confirmed sources — skip the Source preflight and do not pause; the orchestrator ran it once.)
>
> **Coverage — floor, not ceiling.** The grep patterns and site lists in this skill are a **seed, not an exhaustive enumeration**. After running them, widen the search with full reasoning. The techniques here are **illustrative examples, not the allowed set** — use any approach your reasoning suggests, including ones not listed: e.g. semantic search (by behavior/effect, not just token), resolving macros / wrappers / typedefs / indirection the literal pattern can't match, following the call graph to callers and callees, and diffing the WH/BH/QSR variants to catch a site present in one arch and missing in another. If you can find a hazard, primitive, or site the encoded patterns don't cover — by any means — pursue and report it; do **not** clamp a stronger analysis to this list or to these techniques. State any residual coverage gaps explicitly (no silent caps).
>
> **Execution — parallel by default.** When enumeration yields more than a few sites/files, **fan out concurrent `Agent` calls by default** (one per file/subsystem, a fresh context each), saturating the available concurrency (~10–16 at once); go inline only for a trivial set. The per-file fan-out described under *Thoroughness* is the **default**, not an exhaustive-only option. The cross-referencing/synthesis of results stays sequential (it must follow the per-unit findings). The heavyweight **Workflow** tool still requires explicit multi-agent opt-in — it is the opt-in exhaustive tier, not the default. Don't over-spawn a tiny diff.
>
> **Persisting results — single writer, incremental.** Agents only **return** their findings; they never write a shared file (no concurrent-write clobbering). If findings are persisted to a file, the orchestrator/caller is the **sole writer** and **appends each wave's returns as they arrive** — incremental, never only-at-the-end — so an interrupt preserves every completed wave's findings.

## Recall preflight — run the `llk-audit` tool first (augmentor, not a verdict)
Get the deterministic candidate list before manual analysis (it enumerates the
SrcA/SrcB data-valid handshake control points and flags the two ISA-grounded
mechanical patterns):

    tt_metal/tt-llk/.claude/tools/llk-audit/run.sh <wormhole|blackhole|quasar> --checks srcreg-bank
    # from the tt-metal repo root; do NOT cd into the tool dir (run.sh self-locates)
    # PR-scoped: add --changed [BASE] (default main) to report only findings touching a changed file.
    # candidates: out/audit.<arch>.json -> .checks["srcreg-bank"].findings

`RAW_SETDVALID_BH` = a raw `TTI_SETDVALID` on Blackhole (ISA-unsupported — it
corrupts `ImpliedSrcBFmt`; the supported form is `UNPACR_NOP(...,SET_DVALID,...)`);
`DVALID_SET` / `DVALID_CLEAR` = the dvalid handshake control points to place- and
lockstep-check.

`DEST2SRC_NO_MATH_DRAIN` = the **math half of check 5**, flagged mechanically: a
`MOVD2A`/`MOVD2B` whose nearest preceding `STALLWAIT` gates on the Src bank-valid
condition but not on `p_stall::MATH`. `DEST2SRC_WAIT_UNSEEN` (no stall in that
function — the gate may be in the caller or in the MOP that replays the move) and
`DEST2SRC_WAIT_UNRELATED` (a stall gating neither) are recall candidates, as is
`DEST2SRC_NO_MATH_DRAIN_UNCONFIRMED`, which is what Quasar emits instead of the flag
because the mechanism is grounded on the WH/BH bank model only.

All are **candidates**, not verdicts. The tool does NOT model bank-flip lockstep (the
`MOV*2D` consume side), dvalid placement, single-thread ownership, the BH
`DISABLE_IMPLIED_SRC?_FMT` bit, or the Quasar SrcS lane. For check 5 specifically it
reads only the nearest *textually* preceding stall in the *same* function (no control
flow, so a stall inside an `if constexpr` is credited to a move outside it), and it does
**not** pair a publication with its consumer: `DUMMY_PUBLISH_UNGUARDED` recalls the
**unpack half of check 5** — a publication with no wait-like bit, no preceding
`SRCA_CLR`/`SRCB_CLR` stall and no preceding real `UNPACR` — but only inside functions
whose NAME marks them as dummy-bank publishers (`*dummy_valid*`, `*switch_to_reduce*`,
`*reuse_dest*`). A publisher named otherwise (e.g. rmsnorm's MOP-config, which builds
the publication as a `static constexpr` MOP op) is NOT recalled. **Widen** with the
method below for all of those. It never clears a site; you decide. If unbuilt, proceed
manually.

## The bug class (precise)
The backend **data** memories are shared and have their own hardware flow control, distinct from config registers, Tensix semaphores, mailboxes, and CBs:
- **`SrcA` / `SrcB`** — each has **2 banks** carrying an `AllowedClient ∈ {Unpackers, MatrixUnit}`, plus four bank-pointer bits (`MatrixUnit::SrcABank/SrcBBank`, `Unpackers[0/1]::SrcBank`). The unpacker fills a bank and hands it to the FPU (set data-valid); the FPU consumes and hands it back. The **Wait Gate enforces this in hardware**: an FPU instruction stalls until the relevant bank's `AllowedClient == MatrixUnit`; `UNPACR` can start but stalls mid-execution until `AllowedClient` is appropriate. Software must keep the two sides' bank pointers in **lockstep** and place the valid/clear at the right point.
- **`Dst`** and **`LReg`** exist **once** (not per-thread). Threads can overwrite each other's data. The math↔pack `Dst` handoff rides the `MATH_PACK` semaphore (owned by `semaphore-handshake-audit`); cross-thread `LReg` is the (declared-but-unused) `mutex::SFPU` (also that audit). THIS audit owns the parts they don't: bank-flip / dvalid correctness, and any Dst/LReg sharing not mediated by those primitives.

A desync → the FPU reads a bank the unpacker is still filling, or a thread clobbers a live Dst/LReg → **silent data corruption** (rarely deadlock).

## Ground-truth (confirm via tt-isa-docs MCP)
`SrcASrcB.md` (bank model, `AllowedClient`, the four bank bits, BH implied-format-per-bank), `WaitGate.md` (the hardware-enforced `AllowedClient` stall + the `UNPACR` mid-execution wait), `Dst.md`, `LReg.md` (shared-once). Re-read per arch — WH and BH differ (e.g. BH per-bank implied format).

## What to check
1. **Bank-flip lockstep.** Over a complete tile/op, the unpacker's `SrcBank` increments and the FPU's `SrcABank`/`SrcBBank` increments must match 1:1. A conditional `UNPACR`, an op that flips one side but not the other, or a face/tile-count mismatch desyncs them → FPU reads the wrong (still-being-written) bank. Walk every branch.
2. **Valid/clear placement (dvalid handshake).** Data-valid handed to the MatrixUnit only after the unpack of that bank completes; handed back to the unpackers only after the FPU has consumed it. Flag a set-valid before the fill is complete, or a clear/reuse before the FPU drains.
3. **Single-thread ownership of the bank state.** The ISA requires "each relevant backend execution unit is only in use by one thread at a time." Two threads both driving the unpackers, or both issuing FPU ops, corrupt the shared bank-pointer bits. Flag any cross-thread contention on the unpacker or FPU bank state that isn't excluded by a handshake.
4. **Dst/LReg overwrite outside the known primitives.** A raw FPU/SFPU/pack access to `Dst`, or cross-thread `LReg`, that is NOT ordered by `MATH_PACK` / `mutex::SFPU` → flag and hand the semaphore half to `semaphore-handshake-audit`; this audit confirms the data-register access itself.
5. **Dest→Src reuse (`MOVD2A`/`MOVD2B`) — check BOTH sides of the handshake.** This is the one bank handoff where the Wait Gate does **not** protect you, and where fixing one side leaves the other broken. Treat the two as independent findings; never close one on the strength of the other.

   **(a) Math side — source-valid alone is insufficient.** `MOVD2A`/`MOVD2B` *write* Src from Dest, so they sit outside the Wait Gate's automatic `AllowedClient` wait, which covers only FPU instructions that *read* Src (the ISA notes on the source-valid conditions say they are "rarely needed" for exactly that reason — these writes are the exception). `MOVD2A.md` / `MOVD2B.md` say outright that they do not auto-wait and direct software to `STALLWAIT`.

   But a `STALLWAIT` selecting **only** the source-valid condition is still wrong. That condition indexes `MatrixUnit.Src?Bank` **live** at the Wait Gate, and that pointer is advanced in the *epilogue* of the preceding Matrix Unit instruction — see the `if (FlipSrcA)` / `if (FlipSrcB)` block at the end of `ELWMUL.md`'s functional model, and equivalently `SETRWC` with a `CLR_*` operand. With a bank-flipping op still in flight the condition tests the **pre-flip** bank, which the Matrix Unit still owns, is satisfied vacuously, and releases — by the time the move executes the flip has landed and it writes the **post-flip** bank the unpacker owns and may be filling.

   So the wait must **also** select the "this thread has an instruction in any stage of the Matrix Unit (FPU) pipeline" condition (`p_stall::MATH`), draining the pipe so the source-valid test observes the post-flip pointer. That condition's documented precondition is that the block mask blocks new Matrix Unit instructions (`p_stall::STALL_MATH`) — verify that too. Flag any `STALLWAIT` gating a `MOVD2A`/`MOVD2B` whose condition mask carries `SRC?_VLD` without `MATH`. **Symptom is a silent wrong value, never a hang**, because every FPU instruction that *reads* Src does auto-wait — so absence of hangs is not evidence of safety here.

   **(b) Unpack side — the dummy publication must wait on the bank it clears.** These moves depend on the unpacker publishing a dummy DVALID (a `UNPACR_NOP` doing ZEROSRC and/or SET_DVALID) to hand the bank over. That instruction clears/publishes `Unpackers[i].SrcBank` but, in its default form, *waits* on `MatrixUnit.Src?Bank` — a **different bank** once double-buffering reaches steady state — so it can clear a bank it never waited for. Correct forms, all present in-tree; accept any one:
   - the "wait like UNPACR" control bit set, so the instruction gates on `Unpackers[i].SrcBank` (BH exposes it as a `STALLWAIT`-clear operand on `UNPACR_NOP`; WH as a distinct `UNP_ZEROSRC_*` encoding), or
   - an explicit preceding `STALLWAIT` on the **unpacker-owned-bank** conditions (`p_stall::SRCA_CLR` / `SRCB_CLR` — `Src?[Unpackers[i].SrcBank].AllowedClient != Unpackers`), or
   - a preceding real `UNPACR` (not `UNPACR_NOP`), which fills the unpacker's own bank and waits for it, so a following `SET_DVALID` inherits a correct wait by sequencing (`UNPACR_NOP_SETDVALID.md`). Note a preceding **plain `UNPACR_NOP ZEROSRC` is NOT a guard**: it satisfies `SET_DVALID`'s need to inherit *a* wait, but its own wait is on `MatrixUnit.Src?Bank` — the wrong bank — so the inherited wait is wrong too.

   A bare `STALLWAIT` on the unpacker *pipeline* condition (`p_stall::UNPACK`) is **not** one of these — it drains the pipe, it does not establish bank ownership. Diff the arches here: one arch's version of a shared helper is often guarded while the other's is not.

   **(c) The join.** Fixing the math side lengthens the math-thread wait and shifts inter-thread timing, which can *unmask* an unguarded publication on the unpack side. When you flag (a), always state whether (b) holds at the paired publisher, and vice versa.

## Method
1. Enumerate the handshake primitives and bank bookkeeping. **Scan the KERNEL
   layer too, not just canonical tt-llk** — hand-written dvalid/bank/`MOV*2D`
   sequences live in `ttnn/`/`models/` kernels (and in ttnn ops that **vendor
   their own `tt_llk` fork** under `.../kernel_includes/tt_llk/`), which a
   canonical-tt-llk-only search misses:
   ```bash
   # from the repo root
   grep -rInE "SETDVALID|CLEARDVALID|CLEARSRC|set_dvalid|clear_src|Src[AB]?Bank|unpack.*bank|MOV[AB]2D|MOVD2[AB]|TTI_UNPACR|STALLWAIT|get_valid" \
        tt_metal/tt-llk/tt_llk_* tt_metal/hw/inc/api ttnn/cpp models --include=*.h --include=*.cpp 2>/dev/null | grep -v /tests/
   ```
2. Per unpack→math op, pair the unpacker's fill/flip with the FPU's consume/flip; trace the bank pointer on both sides across the tile loop. Confirm lockstep, valid/clear ordering, and single-thread ownership.
3. For Dst/LReg, identify the accessing threads and the mediating primitive (or its absence).

## Verdict
- **Bank pointers lockstep on every path, valid/clear correctly ordered, single owner per unit** → SAFE.
- **Bank-flip desync reachable** (counts diverge on a branch) → CORRUPTION (FPU reads unfilled/over-written bank).
- **dvalid set/cleared at the wrong point** → CORRUPTION or stall.
- **`MOVD2A`/`MOVD2B` gated on source-valid without the FPU-pipeline drain** → CORRUPTION (the move writes the post-flip bank the unpacker owns). Silent wrong values, no hang.
- **Dummy publication that waits on the Matrix-Unit bank rather than the unpacker's own** → CORRUPTION (a published bank can be cleared). Report separately from the math-side verdict even when both are present at the same op.
- **Cross-thread contention on bank state / unmediated Dst|LReg sharing** → RACE (hand the semaphore half to `semaphore-handshake-audit`).
- **Risk only on an experimental/unused path or value-invariant** → LATENT — say so.

## Architecture note
**`STALLWAIT` condition/block bit *values* differ between WH and BH.** The `p_stall::` constants carry the same meanings, but their numeric encodings do not line up, so a condition number read from one arch's `STALLWAIT.md` must never be carried over to the other. Always reason with the named constants and re-derive the bit for the arch under audit from that arch's `ckernel_instr_params.h` plus its own `STALLWAIT.md`. Quoting one arch's condition numbering in a finding that spans both arches is a reporting error even when the fix is right.

WH/BH share the bank model; BH adds per-bank implied data format (`ImpliedSrcAFmt/BFmt`) written by the unpacker — verify the implied-format and the data land in the same bank the FPU will read. **On BH a raw `SETDVALID` is ISA-unsupported** (it corrupts `ImpliedSrcBFmt` to an unpredictable value); the supported form is `UNPACR_NOP(...,SET_DVALID,...)`. Flag a raw `TTI_SETDVALID` on BH, and check the implied-format disable bit
`DISABLE_IMPLIED_SRC?_FMT_Base` on the moves that touch that Src bank — grouped by
**BANK, not by data direction**: **SRCA** for `MOVA2D` (SrcA→Dest) **and** `MOVD2A`
(Dest→SrcA); **SRCB** for `MOVB2D` and `MOVD2B`. The moves differ in DATA direction
(`A2D`/`B2D` read the bank into Dest; `D2A`/`D2B` *write* the bank from Dest — a
bank-fill racing dvalid/bank state), but per the live ISA (`MOVD2A.md`) they BOTH
interact with `ImpliedSrcA/BFmt` on Blackhole — the ISA in fact *recommends* setting
`DISABLE_IMPLIED_SRC?_FMT_Base` for the `D2A`/`D2B` moves (its interaction with the
implied format is ill-specified when the bank is invalid) — so do **not** assume the
Dst→Src moves skip the implied-format check. (Direction grounded in the ISA
`MOVD2A.md`/`MOVA2D.md` titles + the `D2A`/`A2D` mnemonic; `ckernel_ops.h` settles
only existence/encoding — its MOV macros carry no direction comment and share a
parameter list.) Quasar's unpack→dest path has its own semaphores (`UNPACK_TO_DEST` / the QSR semaphore map) plus HW AutoTTSync — confirm the model before extending verdicts.

**Do NOT dismiss a Quasar-specific data lane by analogy to the WH/BH 2-bank SrcA/SrcB model.** Quasar adds a third unpacker / `SrcS` lane (`llk_srcs.h`, `UNPACKER2`): audit its dvalid lifecycle in full — both the **set** (producer, e.g. `UNPACR2`) **and** the **clear/consume** (consumer, e.g. `PACR1`) — and whether the lane's interlock fences (e.g. `*_SRCS_RDY` stall conditions) are actually *invoked*. A fence that is **defined but never used** is itself a finding (the lane is unprotected — safe only while it stays unwired/test-only), not grounds to call the lane SAFE. "It's a separate lane, so it doesn't participate in the SrcA/SrcB handshake" is a hypothesis to verify against the QSR ISA/Confluence and to trace in code — never a closure by analogy.

## Output
For each op/site: `file:line` of the unpacker fill/flip and the FPU consume/flip, bank-pointer lockstep result (per branch), dvalid set/clear placement, single-owner check, Dst/LReg mediation, arch, verdict (SAFE / CORRUPTION / RACE / LATENT) + one-line fix. For a Dest→Src move, report **both** halves of check 5 explicitly — the math-side wait mask and the paired unpack-side publication, each with its own `file:line` and verdict — so a half-fixed handshake is never reported as one finding. End with totals per arch.
