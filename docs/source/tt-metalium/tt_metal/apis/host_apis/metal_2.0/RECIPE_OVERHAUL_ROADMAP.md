# Metal 2.0 Op-Porting Recipe — Overhaul Roadmap

> **Working scaffold — NOT a deliverable, NOT part of the recipe.** This is the plan
> for overhauling the porting/audit recipes, written Claude→Claude. Porter-Claudes
> never see it (they read the recipe itself). If this branch is ever productized,
> exclude this file.
>
> **Provenance:** started 2026-07-03 (updated through 2026-07-05) by Claude (Opus 4.8) with Audrey, on branch
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
- **The invariant must be exceptionless (instruction-consistency is load-bearing).**
  Any "safe" carve-out to "don't modify behavior / just swap syntax" teaches the
  porter the rule is *negotiable* — and a Claude generalizes the negotiation, so the
  next drive-by is bigger and you're back to hero-hacking. No carve-outs: the
  carve-out is the crack. This governs how *every* "don't be a hero" instruction in
  the recipe is written. (Surfaced settling Fix #2's Class-3 question.)

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

## Big change #3 — ComputeHardwareConfig / DataMovementHardwareConfig overhaul (PLACEHOLDER — WIP)

> **PLACEHOLDER — do not act on this yet; the code change is in progress.**
> Code-change owner: **River.** Expected to settle ~2026-07-06 ("a day or so," per
> Audrey, as of 2026-07-05). Flesh out into a real change entry once it lands.

- **What:** overhaul of how `ComputeHardwareConfig` and `DataMovementHardwareConfig`
  work, plus new **TTNN-side helper functions** to populate them more effectively.
- **Why it matters to us:** the recipe must become aware of the new shape — how a
  porter populates these configs (and via which TTNN helpers) will change.
- **Status:** details deliberately deferred until the change stabilizes (discussing
  before then just churns). When it lands, flesh out: what changed, where it lands in
  the recipe, porter guidance.
- **Source when fleshing out:** current state from River (code) / Audrey. The
  `development-docs/2026-06 ComputeHardwareConfig Cleanup and TTNN Helper.md` design
  doc is **stale** (Audrey, 2026-07-05) — do not rely on it.

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

### Fix #2 — TensorAccessor 3rd-arg (`aligned_page_size`) triage

Source: `development-docs/2026-06-24 … TensorAccessor 3rd-arg taxonomy`.
**4 classes** (the note superseded the old 3-class scheme):

1. **Dynamic page size** (varies with row width across cache-reused shapes) →
   **PORT WORK**: set `dynamic_tensor_shape` (the relaxation this exists for; PR #47456).
2. **Redundant fixed value** (`== aligned_page_size`) → **PORT WORK**: drop (true
   no-op — the binding's static `AlignedPageSize` supplies the identical value).
3. **Latent bug** (divergent value, masked today by access pattern / dead path /
   test config) → **GATE to ops** (not urgent).
4. **Live bug** (divergent, reachable in default config) → **GATE to ops** (⚠ urgent).

- **Porter's bright line (settled):** `3rd arg == aligned_page_size` → **drop**; any
  **divergence → do NOT drop; refuse/flag to ops.** The porter never distinguishes
  Class 3 from Class 4 — that's ops' urgency triage, not the porter's action. This
  sidesteps the *error-prone* 3-vs-4 boundary (the note documents the first sweep
  mis-calling it), so a mis-classification can never smuggle a live behavior change
  into a port.
- **Why hands-off, not drop-to-fix:** dropping a divergent arg fixes a latent bug =
  a behavior change = violates the "just swap syntax" invariant *and* big-change-#2's
  preserve-don't-fix. And it's **structurally enforced**: Metal 2.0 has **no
  divergent-fixed-page mechanism** (deliberately un-built — every divergent fixed
  page found was a bug; analysis + Borys), so a porter *cannot express* one → hitting
  one forces a stop → refuse → ops. (Dream-hammer shape: always-a-bug ⇒
  always-inexpressible.)
- **Operationalize (parallels Diego's spreadsheet):** check in the **taxonomy table
  as a defer-to lookup** (dated, stamped "authoritative until superseded"; auditors
  look up the op, ops front-run the Class-4 list now) **+ bake the classification
  *procedure* into the audit recipe** (the note's "How to classify a new op" +
  discriminators: `tt::tile_size` vs `element_size()*1024` block-float exponent
  section; raw vs aligned page; `page_id>0` reachability; align on strictest arch =
  BH/Quasar 64) as the **staleness backstop** **+ feedback loop** (auditor derives an
  unlisted op → records → table updated; the loop is the table's only maintenance —
  it has no dedicated owner).
- **Lands in:** audit **TensorAccessor-handling** subject (rides the existing
  accessor-site walk), the checked-in taxonomy table, the audit's classification
  procedure. Routing reuses the existing finding-roles (PORT WORK / GATE-to-owner).

### Fix #3 — Self-loop DFB consistency (DM-never / compute-fine / packer→self-loop / INTRA purge)

Cleaning up self-loop guidance across recipe + audit + flowchart (Audrey changed it
more than once; goal = consistency). The recipe's core rule turned out already
correct — the flowchart was the stray. Parts:

- **Ground truth — state consistently + prominently, on the DM-vs-compute *axis*
  (never bare "self-loops OK"):**
  - **Compute-kernel self-loop** — common, necessary (LLKs rely on it), supported on
    ALL arches. Fine.
  - **DM-kernel self-loop** — legal on Gen1 (WH/BH), **illegal on Gen2/Quasar**
    (gen-aware legality landed, `c238a58`). **For the recipe: NEVER use one.** It's a
    **silent Quasar trap** — Gen1 tests *and* the WH legality check both go green, so
    a locally-reasoning porter sees "legal + tests pass → fine" and it only detonates
    at the Quasar port, which is the whole point of porting now. Purpose-driven, not
    legality-driven → **attach the why**, or a porter overrides it.
- **Packer / producer-only DFB → self-loop the unused consumer on the compute kernel
  (Option A; = current recipe).** A compute packer (produces into resident output,
  nothing drains) + LLKs ⇒ must be a DFB; Metal 2.0 rejects single-ended
  producer-only DFBs, so a "pointless" consumer binding is required. Self-loop it —
  do **not** bind a separate kernel (e.g. the writer DM). Why (so it isn't
  re-litigated):
  - Binding a non-draining *separate* kernel as consumer of a *real* producer is the
    exact **deadlock-risky fabricated-consumer** shape the audit already GATEs; the
    compute self-loop is the sanctioned-safe fabrication.
  - "Bind the DM writer" (considered + rejected) doesn't cut factory if-elses: the
    DDR-output vs L1-output configs already differ in DFB *backing* (regular vs
    borrowed-mem), so the factory branches on config regardless — the consumer choice
    rides that existing branch. Self-loop is fully compatible with a clean
    single-factory config-switch.
  - Avoids a false producer→consumer topology edge, and the real case where the op
    has no DM writer at all (confirmed via Borys).
- **INTRA/INTER purge.** The `DFBSelfLoopConnectivity {INTRA, INTER}` API option is
  **removed from the API surface** (pointless placeholder; INTER never supported;
  INTRA now auto-applied in Almeet's backend). No dead API-option refs exist in the
  recipe — the only traces are "INTRA" as backend-lowering *shorthand* (~7 sites,
  mostly `metal2_port_patterns.md`). **Strip the word "INTRA"; keep the mechanism it
  grounded, rephrased plainly; trim deep backend detail** (risc_mask,
  `dataflow_buffer.cpp` cite — not porter-actionable). Folds into the same self-loop
  section rewrite.
- **"Why" mechanism — do NOT assert one** (`[OPEN]`, see Open threads). Whether
  compute self-loops lower due to inter-TRISC sync or because LLK→LLK streaming
  handoff needs a self-loop is unresolved. Phrase at confidence level ("compute
  self-loops are a supported, LLK-relied-on pattern; DM self-loops have no backend
  lowering") or verify with LLK/Almeet owners.
- **Flowchart correction (Audrey's action):** the CB→DFB flowchart's packer note said
  "use the writer DM kernel (if present), otherwise self-loop" — the Option-B stray.
  Audrey updating it to plain self-loop. Until then, trust the recipe over the
  flowchart here.
- **Lands in:** `metal2_port_patterns.md` (self-loop / sync-free / single-ended /
  packer sections — the INTRA sites + the packer rule), audit SPSC single-ended fork,
  recipe DFB-spec self-loop mentions. Flowchart (Audrey).

## Phase-2 style recipe (deferred — separate step, NOT port work)

Metal 2.0-enabled style improvements, run as a distinct pass *after* the initial
minimal port. Bundling any of these into a port would violate scope discipline.
(When this list grows past a handful, split it into its own roadmap doc.)

- **CENTERPIECE — optionally-bound resources (tensor + DFB, also semaphore) →
  first-class kernel arguments.** Today the recipe (**rule 6**) handles a
  conditionally-bound resource with a **CTA→`#define` + `#ifdef`-gating** workaround:
  the selecting condition moves to a kernel-side `#define`, and the binding token's
  `constexpr` alias (and every expression referencing it) is `#ifdef`-gated — needed
  because an unbound token's alias would otherwise fail name lookup. Yucky and
  error-prone (a prior porter deleted comment blocks while adding `#ifdef`s).
  **First-class / "1st-world" kernel arguments** eliminate it: presence/absence
  becomes part of the typed kernel-arg interface (compiler-handled, no preprocessor),
  so the kernel reads a clean conditional instead of `#ifdef` walls. Deferred because
  converting an op to first-class kernel args is a **big lift + big diff** — wrong for
  the minimal initial port, but the phase-2 headline.
  - **Scope-discipline guard (important):** keep the "this is interim pending
    first-class kernel args" knowledge *here in the roadmap, NOT in the porter-facing
    phase-1 recipe.* Telling a porter "the real fix is first-class kernel args" tempts
    the big conversion mid-port — the exact scope-creep the exceptionless-invariant
    guards against. Phase-1 rule 6 stays a plain "do the `#ifdef` workaround, full
    stop."
  - **Verify when writing it:** confirm "first-class kernel arguments" actually covers
    conditional *bindings* (`dfb::`/`ta::`/`sem::`), not just *args* (`args::`). Lead:
    Mo's kernel-arg work (#46623) — but that was CTA/RTA/CRTA *arguments*; bindings are
    a distinct channel today, so confirm the mechanism extends to them.
- **LocalTensorAccessor `[]`-operator upgrade** (Fix #1's Option B) — Case-2a raw-base
  → idiomatic `[]` access.
- *(more from Audrey — less critical than the above; capture as they surface)*

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
- `[OPEN]` **Compute-vs-DM self-loop "why" mechanism** (Fix #3) — inter-TRISC sync vs
  LLK→LLK streaming-handoff-needs-self-loop. Unresolved; don't assert in the recipe.
  Verify with LLK/Almeet owners, else phrase at confidence level.
- `[OPEN]` **Reorg execution** — human/AI-facing directory split + flowchart formats
  (SVG for check-in, PNG for Claude reading). After big changes settle.
- `[OPEN]` **Reference-table mechanics** — Diego's factory CSV *and* the 3rd-arg
  taxonomy table (Fix #2): vet + curate + stamp + check-in; co-locate the two
  defer-to tables. Both pair with a procedure/tripwire backstop + feedback loop.
- `[OPEN]` **This doc's fate** — on-branch (chosen, for handoff-locality); exclude
  from any productization.

## File-map (what to read / touch)

| Change | Read first | Touch |
|---|---|---|
| #1 CB handling | `metal2_port_patterns.md`, audit TensorAccessor-handling + SPSC, recipe rule 5 | patterns catalog, audit (2 subjects), recipe rule 5 |
| #2 factory shapes | audit TTNN-factory-concept-analysis + Appendix A, `port_op_to_metal2_ttnn_factory.md` | audit factory-analysis, Appendix A, recipe factory framing, new CSV |
| #3 HW-config overhaul (WIP) | TBD — blocked on River's code change; current state from River/Audrey (the 2026-06 design doc is stale) | TBD |
| Fix #1 LocalTensorAccessor | recipe rule 5, audit TensorAccessor-handling, Dropped-Plumbing Case-2 | same three |
| Fix #2 3rd-arg triage | `development-docs/2026-06-24 … taxonomy`, audit TensorAccessor-handling | audit TensorAccessor-handling, checked-in taxonomy table, audit classification procedure |
| Fix #3 self-loop consistency | `metal2_port_patterns.md` (self-loop/sync-free/packer + INTRA sites), audit SPSC single-ended fork | patterns catalog, audit SPSC fork, recipe DFB-spec self-loop mentions; flowchart (Audrey) |
