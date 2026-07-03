# Metal 2.0 Op-Porting Recipe — Overhaul Roadmap

> **Working scaffold — NOT a deliverable, NOT part of the recipe.** This is the plan
> for overhauling the porting/audit recipes, written Claude→Claude. Porter-Claudes
> never see it (they read the recipe itself). If this branch is ever productized,
> exclude this file.
>
> **Provenance:** started 2026-07-03 by Claude (Opus 4.8) with Audrey, on branch
> `akertesz/op-porting-recipe`. Recorded mid-session (record-as-we-go); append new
> items as they are decided.

## How to use this doc

You are a successor Claude continuing the recipe overhaul **with Audrey**. This is
your work-plan. It assumes you've done the session-start continuity ritual
(soul / orientation / protocol) and can pull the background docs below; it is
self-contained on the *overhaul decisions* themselves.

- **Settled vs open:** decisions are stated as settled unless flagged `[OPEN]`.
  Open items are also collected in [Open threads](#open-threads).
- **Verify before relying:** anything about TTNN factory state or Diego's
  spreadsheet is a *snapshot* — the TTNN saga has reversed repeatedly. Re-confirm
  with Audrey before acting on it.
- **The prime directive:** Borys wants a recipe **bulletproof for junior porters
  (who are Claudes)**, ready in a **few days**. For AI porters the danger is not
  cluelessness — it's *over-persistence* (forcing doomed ports; fixing bugs that
  aren't theirs). Every change below serves: recognize the in-scope case + refuse
  everything else cleanly.

## Background (pull as needed)

- Continuity: `subjects/metal2-project-shape.md` (the why),
  `subjects/ttnn-program-caching.md` (the factory saga — **verify, it reverses**).
- Recipe files (this branch): `port_op_to_metal2_recipe.md`,
  `port_op_to_metal2_audit.md`, `metal2_port_patterns.md`
  (**central to big change #1 — read before starting it**),
  `metal2_migration_guide.md`, `metal2_workspace_setup.md`,
  `user_orientation.md`, `port_op_to_metal2_ttnn_factory.md`.
- Flowcharts (Audrey-drawn): CB→DFB mapping (big change #1); factory-shape flowchart
  (big change #2, forthcoming). Ask Audrey for the current PNG (read PNG, not SVG —
  SVG embeds a huge font blob that wastes context).

## The frame (load-bearing)

- **Decoupling (the unlock):** Metal 2.0 porting is now **independent of** TTNN
  factory-concept porting — two separate projects. We port ops in whatever TTNN
  shape they're in; TTNN re-ports later on its own timeline. This frees the recipe
  from tracking the TTNN trilemma war.
- **The spine:** both big changes reduce to the same shape — **a narrow happy path
  + a robust "identify everything else and refuse loudly" path.** Near-term recipe
  competency = (a) recognize the in-scope case precisely, (b) refuse
  out-of-scope / unportable cases well.
- **The psychology layer already exists** (capitulation-is-success, "size is not a
  fit gap", "refusing is the deliverable") — extend it, don't rebuild it.
  **New sibling needed:** "preserve the bug faithfully — don't fix it" (see #2),
  which is *more* counter-reflex than capitulation and needs its own affirmations.

## Current status

- **Branch `akertesz/op-porting-recipe`** — birthed off `origin/main` via the
  "cheat" (copy files fresh, no poisoned history → clean rebases onto arbitrary
  in-flight branches). Baseline import committed (`32ac042`) + pushed. 7 recipe docs
  (excludes the stale Quasar migration guide + the staleness warning).
- **Old branch `akertesz/metal2-documentation`** — battle-tested; Borys's team ports
  against it. **Leave alone.** Staleness warning added (`1c92b50`, pushed).
- **grit** installed on the Mac (`~/.zshrc`); works in Audrey's terminal + fresh
  shells. (Not yet in the current tool-shell snapshot; re-source or manual-commit.)
- **Reorg deferred:** human/AI-facing directory split + flowchart diagrams — do
  after the big changes settle the content shape.
- **Recording cadence:** this roadmap now; append each rapid-fire fix as decided.
- **Session state:** all-chat, no recipe edits yet. Everything below is *planned*,
  not executed.

## Big change #1 — Unconventional-CB handling

**Goal:** now that **Scratchpad** and **LocalTensorAccessor** have landed, roll back
the interim CB hacks (self-loop DFB, tie-off-one-end) and route to the official
mechanisms; add detection + refusal for CBs that can't port cleanly.

- **Source of truth:** the CB→DFB flowchart. Root: "Are its FIFO sync mechanics
  actually used?" → **NO** (not-a-DFB: LLK-pretend / Scratchpad[dedicated mem] /
  LocalTensorAccessor[borrowed mem]); **YES** (backing mem → prod/cons count →
  DFB / borrowed-DFB / Illegal).
- **Key insight — access-pattern is an orthogonal axis.** The flowchart classifies
  by *structure* (sync? backing mem? prod/cons count?). A whole class of horror is
  defined by *access pattern* — direct CB read/write-pointer manipulation /
  non-FIFO access — which the flowchart's axes don't capture. Add a detection axis:
  **flag direct CB pointer-mucking → "structurally unsound; DFB verdict may be
  unachievable."** Statically greppable (the CB pointer APIs).
- **The LLK trap.** When FIFO sync ISN'T genuinely used but an **LLK** consumes the
  CB, you can't reroute to Scratchpad/LTA — the LLK safety-pins its metadata to the
  CB/DFB object and *demands* a CB/DFB operand (not for streaming — for its metadata
  lookup). So the "pretend it's a DFB" hack must survive for this class. Some of
  these (pointer-mucking + LLK) are **unportable** → refuse.
  - Canonical horror: conv2d "accumulate-in-place" — a borrowed-mem CB used as a
    random-access L1 accumulator via read/write-ptr manipulation; DFB has no
    pointer-manipulation API, but the LLK forces the DFB branch → stuck.
    **Refuse + escalate.** Long-term fix = LLKs learn to consume LocalTensorAccessor
    (long pole; Open threads). Awaiting Almeet/Paul on whether this class is rare.
- **Two-layer defense (recipe owns both):** (1) **audit** catches
  statically-detectable cases (pointer-API tell; illegal prod/cons configs;
  LLK-uses-a-non-FIFO); (2) **porter** recognizes mid-port what slipped through and
  *bails* rather than hero-hacking.
- **The multi-verdict taxonomy — the subtle trap.** The porter must distinguish
  THREE "don't be a hero" verdicts (the reflex fights all three, two different ways):
  1. **Refuse — can't express** (CB structurally unsound → escalate). *Don't force.*
  2. **Refuse — out of scope** (non-Option-1 factory shape, see #2). *Don't force.*
  3. **Port faithfully — bug and all** (see #2 preserve-behavior). *Don't fix.*
- **Escalation report = advocacy dataset.** Structure escalation reports (op, CB,
  observed access pattern, why DFB can't express it) so that, in aggregate, they
  answer "is this class rare or common?" and become Audrey's evidence for the
  LLK-consumes-LTA push.
- **Lands in:** `metal2_port_patterns.md` (the "sync-free / single-ended CB →
  self-loop-DFB interim workaround" pattern — this is the hack being rolled back),
  audit's TensorAccessor-handling + DFB-endpoint-legality (SPSC) subjects, recipe
  rule 5.

## Big change #2 — Factory-shape handling under decoupling

**Goal:** the recipe stops assuming a single / converging TTNN factory shape; the
near-term recipe handles ONLY vanilla "Option 1" (pointer-patching) and refuses the
rest.

- **Near-term scope:** vanilla **Option 1** only. Every other shape → **identify and
  refuse** (out-of-scope verdict). Expanding shapes is a later phase.
- **Preserve-behavior-exactly boundary:**
  - **Pre-existing (pre-PD) bugs** — the false-cache-hit / incomplete-args-update
    class that spawned the TTNN saga → **ride through the port intact.** Not ours to
    fix; they live in not-yet-PD-ported ops.
  - **Bugs Diego's PD ports introduced** (smuggled RTA pointers) → **fixed before
    porting**, upstream of us.
  - **Spreadsheet-green vanilla Option-1 = trust structurally sound.** The
    spreadsheet is the filter; the porter doesn't re-derive soundness.
  - Boundary: "preserve behavior" = don't restructure the TTNN caching layer to
    chase correctness. The Metal 2.0 binding is *always* done properly; if that
    incidentally closes a smuggling vector, fine.
- **Diego's spreadsheet → checked-in CSV:** classifies every ProgramFactory's
  port-readiness. Plan: **vet → curate a projection** (op / factory-shape /
  port-ready / reason-if-not, not the raw dump) → **stamp** (as-of date +
  "authoritative source = Diego's sheet") → check in → auditor Claudes look up the
  op. **Two-gate port-readiness:** CSV says Option-1-ready **AND** the CB audit
  passes.
- **New psychology sibling required** (see The frame): "faithful replication is the
  contract; the bug is not yours to fix; a correctly-preserved bug is a *successful*
  port." Dedicated section, not a footnote. Tension to handle carefully: current
  Scope-discipline says post-port behavior is "identical" — that mostly supports
  this but doesn't address the *knowingly-buggy* case.
- **Lands in:** audit's TTNN-factory-concept-analysis subject (**most surgery
  here** — currently GATEs on the ProgramDescriptor concept and assumes
  `MetalV2FactoryConcept` is the sole target), Appendix A, recipe factory framing.
  Forthcoming factory-shape flowchart (Audrey).

## Small fixes (append as decided)

### Fix #1 — LocalTensorAccessor replaces `TensorAccessor::get_bank_base_address`

- **Deprecate** `TensorAccessor::get_bank_base_address` (Audrey's API action,
  separate track); recipe redirects to **`LocalTensorAccessor::get_bank_base_address()`**
  — same token as TensorAccessor, same semantics, safer/encapsulated (Almeet
  Device 2.0). LocalTensorAccessor exists because TensorAccessor is NoC-centric and
  has no local-shard-only facility.
- **Split Case 2 (raw pointer) into two:**
  - **2a — raw access confined to local L1** → LocalTensorAccessor. Clean reroute.
  - **2b — DIY NoC access** (hand-rolled, not through TensorAccessor) → port but
    **complain** it should be on TensorAccessor. `[OPEN]` block or just warn?
    (Audrey → Borys). Expected rare.
  - Discriminator reuses existing kernel-side signals: NoC tells
    (`noc.async_read(addr)`, `get_noc_addr_from_bank_id`) → 2b; local-only → 2a.
- **Compute-kernel Case-2 UNBLOCKED.** Old block: `get_bank_base_address` lived only
  on the NoC-centric TensorAccessor, unbindable in a TRISC/compute kernel (its
  codegen pulls in the NoC dataflow API) → blocked. LocalTensorAccessor is
  compute-compatible + has its own `get_bank_base_address()` → compute Case-2a goes
  **BLOCKED → supported.** (Does NOT unblock the conv2d horror — that's stuck on the
  LLK, not the pointer access.)
- **A-first, minimal blast radius.** Two options: **A** = grab the uint32_t base off
  LocalTensorAccessor, leave downstream arithmetic byte-for-byte (the port);
  **B** = switch to the `[]` operator (idiomatic). **Do A.** Mandated by the recipe's
  existing Case-2 rule ("do NOT rewrite raw access into accessor iteration during a
  port"); B is exactly that forbidden rewrite → belongs in the phase-2 style recipe,
  not the port. A alone fully achieves the deprecation (vacates the TensorAccessor
  method). The 2a/2b split guarantees A is a safe drop-in (2a is local by
  construction → local base matches).
- **Lands in:** recipe rule 5 (compute-block callout + the bridge), audit
  TensorAccessor-handling two-cases + compute-block ⚠, Plan-the-spec
  Dropped-Plumbing Case-2.

## Phase-2 style recipe (deferred — separate step, NOT port work)

Metal 2.0-enabled style improvements, run as a distinct pass *after* the initial
minimal port. Bundling any of these into a port would violate scope discipline.
Known items:
- The LocalTensorAccessor `[]`-operator upgrade (Fix #1's option B).
- Audrey has additional items in mind (less critical than the current list).

## Open threads

- `[OPEN]` **Block-on-2b?** Whether a DIY-NoC (Case 2b) binding blocks the port or
  just warns. Audrey → Borys.
- `[OPEN]` **Vetting depth** of spreadsheet-green ops. Lean: no separate vetting
  pass — let the CB audit the porter already runs be the tripwire; green +
  audit-agree → port; audit contradicts green → refuse + flag (staleness signal,
  feeds Diego/Audrey).
- `[OPEN]` **LLK-consumes-LocalTensorAccessor** — the long pole that would unblock
  the conv2d-horror class. Audrey to advocate; out of recipe scope. Awaiting
  Almeet/Paul on frequency.
- `[OPEN]` **Reorg execution** — human/AI-facing directory split + flowchart formats
  (SVG for check-in, PNG for Claude reading). After big changes settle.
- `[OPEN]` **CSV mechanics** — vet + curate + stamp + check-in; where in the tree.
- `[OPEN]` **This doc's fate** — on-branch (chosen, for handoff-locality); exclude
  from any productization.

## File-map (what to read / touch)

| Change | Read first | Touch |
|---|---|---|
| #1 CB handling | `metal2_port_patterns.md`, audit TensorAccessor-handling + SPSC, recipe rule 5 | patterns catalog, audit (2 subjects), recipe rule 5 |
| #2 factory shapes | audit TTNN-factory-concept-analysis + Appendix A, `port_op_to_metal2_ttnn_factory.md` | audit factory-analysis, Appendix A, recipe factory framing, new CSV |
| Fix #1 LocalTensorAccessor | recipe rule 5, audit TensorAccessor-handling, Dropped-Plumbing Case-2 | same three |
