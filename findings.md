# RMS Norm — Agent-Pipeline Findings, Footguns & Prompt Gaps

A living retrospective on the **agentic development pipeline** (incremental-planner →
implementer → verifier → refinement loop, plus expert-debugger / static-analyzer) as
exercised on the `rms_norm` op. This tracks bugs the agents hit, footguns they walked
into, where a recorded root-cause turned out to be imprecise, and — most importantly —
**what an agent would have needed to know (and where that knowledge should live) to
avoid the footgun**.

Scope note: many findings here are about the **eval system / prompt surface**, not about
`rms_norm` specifically. They are filed here because this run surfaced them; several
likely generalize to every op the pipeline produces.

Status legend: ✅ verified from source · 🟡 plausible, not yet deep-verified · ⬜ open question

---

## Finding 1 — Multi-consumer CB race in Regime B (the `cb_partial` bug) ✅

### What the bug was
Regime B (cross-core W-split allgather) produced numerically wrong output on wide-W
shapes — a near-uniform scale error (~0.5× at W=4096/8192, ~0.26× at W=16384; and a
~0.978× per-row variant fixed later in Refinement 1b). PCC stayed ~0.9998 because the
error was a pure scale, so PCC-gated acceptance never caught it.

### Root cause (corrected — see "Narrative correction" below)
`cb_partial` (idx 25, 2 pages) was used by **two kernels at once**:
- **compute** — as the in-place accumulator for the pass-1 chunk reduce *and* the
  Regime B K-partial combine (`add<cb_partial, cb_external, cb_partial>`), plus finalize.
- **reader_b** — as the **mcast source**: `cb_wait_front(cb_partial,1)` →
  `get_read_ptr(cb_partial)` → mcast send → `cb_pop_front(cb_partial,1)`
  (`kernels/rms_norm_reader_b.cpp:148,149,167`).

Two independent RISCs advancing the same CB's read pointer / `avail` counter corrupts the
FIFO accounting: the reader pops a page compute still believes it owns, the combine folds
a stale/wrong page → wrong `mean(x²)` → scale error. **This is a textbook single-consumer
violation.**

### The subtlety we corrected (important)
The fix commit (`35ecd65c22` / `3926ff461d`) frames the root cause as a *page-aliasing
timing race*: "on a 2-page CB, pop-front then reserve-back land on different physical
pages, so under fast dispatch the next unpack reads a page the previous pack hasn't
drained — every other term is silently dropped." We traced the actual `reduce`+`Accumulate`
FIFO pointer logic (`reduce_helpers_compute.inl:168-191, 411-419`) by hand:

- On a **thread-local** CB, `avail` oscillates 0↔1, so every `wait_front` reads exactly
  the page the previous iteration pushed. `push_back`/`wait_front` establish a real
  happens-before across the compute TRISCs (pack writes L1, *then* increments the
  semaphore the unpacker waits on). **A 2-page in-place accumulate is therefore correct
  as long as the CB has exactly one producer and one consumer.** FIFO correctness holds
  for *any* page count ≥ 1.
- The load-bearing part of the fix is **un-sharing** (`cb_partial` became a clean
  compute→reader handoff; accumulation moved to the compute-private `cb_combine`), **not**
  the reduction to a single page. A compute-private 2-page accumulator would also have
  been correct.

**Two accumulation "sites" from the commit, re-classified:**
- **Site 1 (Regime B combine)** — ✅ genuine bug: shared CB (compute + reader). Real.
- **Site 2 (pass-1 chunk accumulate, thread-local)** — ⬜ structurally sound on 2 pages by
  our trace; its attributed "every-other-chunk drop" is *not* explained by the FIFO logic.
  Either a genuine hardware pack→unpack visibility hazard the semaphore FIFO doesn't model
  (fixed by the 1-page blocking `reserve_back` as a side effect), or a misattribution that
  rode along with the Site-1 fix. **A minimal probe (Accumulate fold of all-ones × N into a
  2-page vs 1-page thread-local CB, under fast dispatch) would settle it. Not yet run.**

### Narrative correction (do not propagate the changelog wording as fact)
The changelog / commit messages describe this as a "2-page in-place accumulator
page-aliasing race." That framing is imprecise: the real, verifiable defect is a
**single-consumer violation** (compute + reader on one CB). The "single-page forces
pop-before-reserve" mechanism is not what was broken for a thread-local CB. Anyone reading
the changelog later should treat the "timing race / every-other-term" explanation as the
debugger agent's hypothesis, not established fact.

### Which agent introduced it, and why the pipeline didn't catch it
- **Introduced by the planner.** `op_design.md`'s CB table lists `cb_partial_local` (c_25)
  with **Consumer = "compute (finalize) *and* reader (mcast source, Regime B)"** — two
  consumers, written explicitly. The design's "CB sync verification" even signs off on it:
  *"reduce pushes 1; finalize (and Regime-B reader) wait 1."* It satisfied the only CB rule
  the planner is given (push == wait) while encoding a two-consumer CB.
- **Implementer** faithfully built the design (the reader wraps the mcast in raw
  `cb_wait_front`/`cb_pop_front` on `cb_partial` exactly as the design's API-mapping table
  instructed).
- **Detection was late and misrouted:** it first showed only as a wide-W numeric error,
  which the Phase-0 verifier filed as a **bf16 precision** problem (see Finding 2). It took
  the expert-debugger, an all-ones repro, and DEVICE_PRINT to reach the CB.

### The prompt gap (root cause of the root cause) ✅
The single-producer/single-consumer rule is **absent from the planner's entire surface**:
- Planner prompt `.claude/agents/incremental-planner.md` §4 "Design CB Layout" + the
  design checklist state only: *"CB sync rule: Producer push count MUST equal consumer
  wait count."* No SPSC.
- Planner's referenced CB doc `.claude/references/ttnn-cb-memory-fundamentals.md`
  §"CB Synchronization Invariant" (billed "THE #1 CAUSE OF KERNEL HANGS") is *only* the
  push==wait rule; its example even shows a lone reader + lone compute, quietly assuming
  single-producer/single-consumer without ever stating it as a constraint. A grep for any
  SPSC phrasing in that file returns nothing.
- Meanwhile SPSC is stated forcefully on **every downstream surface**: `debug-ttnn-op`
  SKILL §"Single producer, single consumer" ("use two separate CBs… CBs are cheap"),
  `memory-layouts` SKILL §2.2/§2.4 (§2.4 *explicitly* says in-place eltwise where output
  CB == input CB counts as a consumer, so "if the writer also consumes the same CB, that's
  two consumers"), `memory-budget-metal` SKILL, `static-analysis-checklist.md` §4.1
  ("read/write pointers do not support concurrent access from multiple threads"), and the
  `ttnn-static-analyzer` agent.

**So the one agent whose job is to lay out CBs is the one agent not told the CB
single-consumer rule.** It designed a two-consumer CB, checked it against push==wait, and
passed its own gate.

### What better understanding would have prevented it
1. **Put the SPSC rule where CBs are *designed*, not just where they're debugged.** Add to
   the planner prompt §4 and to `ttnn-cb-memory-fundamentals.md`: *"Every CB has exactly
   one producer kernel and one consumer kernel over its lifetime. A CB that is both an
   in-place accumulator and a cross-kernel handoff has two consumers — split it. push==wait
   is necessary but NOT sufficient."*
2. **Make the planner's CB table schema single-valued.** If the Producer/Consumer columns
   are constrained to one kernel each, a two-consumer entry becomes structurally
   un-writable — the design can't express the bug. (Here the table literally had the word
   "and" in the Consumer cell.)
3. **Cross-kernel-shared CBs (mcast handoff, allgather sources) deserve an explicit design
   callout** — they are the main way a "one producer / one consumer" CB accidentally grows
   a second consumer. The `mcast_pipe` pattern in particular hands a raw L1 read-ptr to the
   reader; the design should require a *dedicated* handoff CB distinct from any accumulator.
4. **Precision vs scale triage for the verifier** (see Finding 2): a high-PCC + high-RMS +
   uniform-ratio signature is a *scale/structural* bug, not rounding — route it to the
   debugger, not to "add fp32 intermediates."

---

## Finding 2 — Wide-W failure misdiagnosed as precision by the Phase-0 verifier 🟡

The Phase-0 verifier attributed the wide-W failures to the bf16 `cb_xsq` intermediate
rounding each x² before the fp32 reduce, and recommended fp32 intermediates (Refinement 1).
There *was* a real precision component (fp32 `cb_xsq` did help small effects, and the fp32
input path later needed the SFPU square in 1b), but the **dominant** wide-W error was the
Finding-1 CB race, which is scale-structural, not rounding. Signature that should have
flagged it: PCC ~0.9998 (very high) *with* rel-RMS ~0.49 (very high) *and* an
output/reference ratio clustered near a constant (~0.5) → uniform scale, not noise.

**Better understanding:** the verifier/precision-baseline tooling should compute and
surface the got/true **ratio distribution**, and encode the heuristic "high-PCC +
high-RMS + tight ratio cluster ⇒ scale/structural bug, not precision." This is partially
captured in `debug-ttnn-op`'s value-pattern table but was not applied at verify time.
(Corroborated by the existing memory note `rms-norm-phase0-regimeB-race-misdiagnosis`.)

---

## Finding 3 — Grid-dependent regime selection masked the bug locally 🟡

`ttnn.open_device` reports an 8×7 = 56-core grid → wide-W R==1 routes to **Regime A**
(correct path); the eval harness device reports 8×8 = 64 → routes to **Regime B** (the
buggy path). So the bug "passed" under `open_device` and failed under the harness, which
read as device flakiness rather than a code path difference. (See memory
`device-open-idiom-grid-disparity`.)

**Better understanding:** any agent validating a multi-regime op must pin/log which regime
a shape actually selects on the live grid (the Refinement tests later added exactly this —
"assert Regime B selection or skip"). A design that branches on `grid` should state the
selection function explicitly and require regime-pinned tests. This should be a planner
deliverable for any op with >1 compute regime.

---

## Finding 4 — "cross-test L1 fragmentation" as a catch-all for red golden cells 🟡

Nearly every refinement reported dozens–hundreds of `supported_fail` cells as
"cross-test L1 fragmentation" (`program.cpp:1741 Statically allocated CBs clash with L1`),
asserted to pass in isolation and therefore not op defects. This may well be true, but it
is repeated as a self-diagnosis without independent audit, and it conveniently absorbs any
red cell. ⬜ Worth spot-checking a sample: does the harness actually leak sharded L1 across
module-scoped-device cells, and would `ttnn.deallocate` on the rejection path clear them?

**Better understanding:** the harness should either deallocate rejected sharded inputs, or
the verifier should distinguish "environmental fragmentation" from "op OOM" mechanically
(e.g. re-run the failing cell in a fresh device) rather than by narrative.

---

## Finding 5 — Unresolved structural blocker escalated correctly (5d) ✅ (process win)

RM input + TILE gamma + WIDTH/BLOCK sharded is genuinely blocked: a sub-tile RM shard needs
a sub-32 column range of a 32-wide gamma TILE, and there is no in-kernel primitive to
extract it; sub-tile-ness is shape-derived, so it can't be a clean EXCLUSION cell. The
implementer correctly (a) narrowed the EXCLUSION rather than over-claiming, and (b)
escalated the Path-1 (mark INVALID in feature_spec — forbidden for the implementer to edit)
vs Path-2 (on-device gamma repack — risky) decision to the user instead of guessing. This
is the pipeline behaving well under a real constraint; noted as a positive pattern.

---

## Finding 6 — Refinement 2 (non-aligned shapes): tile-geometry alignment footgun ✅

R2 (add `w_non_aligned` / `h_non_aligned` to `SUPPORTED["alignment"]`) was clean relative
to 1/1b, but surfaced two defects of the **same class** plus a couple of process notes.

### 6a. Two Phase-0 geometry formulas silently assumed tile-alignment ✅
Both tile-count formulas were written in Phase 0 for tile-aligned shapes and broke only
when R2 made alignment a variable:
- **H (the real bug — `PCC = nan`):** `R = (volume // W) // TILE_DIM` (flat-row count).
  In TILE layout each `(N,C,H,W)` image is tile-padded *independently*, so the physical
  grid is `(batch, Ht, Wt)` with `Ht = ceil(H/32)`, and the correct tile-row count is
  `batch * Ht`, not `floor(batch*H / 32)`. For `4x8x47x256`: wrong `R = 47` vs correct
  `R = 64`. Two failure modes combine: **17 tile-rows never processed**, *and* the flat
  enumeration computes the wrong tile-index stride, so images past the first read
  **misaligned physical tiles** → garbage into the reduce → `rsqrt` of a bad `mean(x²)`
  → `nan`. Fix: `R = batch * ceil(H/32)` (`rms_norm_program_descriptor.py`).
- **W (latent):** `Wt = W // TILE_DIM` silently dropped the final partial W-tile.
  Harmless while only aligned W was supported; fixed to `ceil(W/32)`.

**Contrast with Finding 1:** this was a **loud** failure (`nan`, caught instantly by a PCC
test), the opposite of R1's PCC-invisible scale bug. The well-tested masking path held; the
untested index-count path slipped through — the right way round.

**Better understanding / fix:** derive tile geometry as alignment-aware from Phase 0
(`ceil`, per-image `batch*Ht`) even when only aligned shapes are exercised, or route it
through a shared geometry helper / the `padding-in-ttnn` skill. "Correct in the common
(aligned) case, wrong at the boundary" is the trap; the boundary is exactly what a later
alignment refinement will hit.

### 6b. The W-masking method: multiply-by-zero in the scaler, not input zeroing ✅
Masking is done in the **scaler**, applied multiplicatively during the reduce — the input's
garbage padding is never touched. `reduce<SUM, REDUCE_ROW>` uses the **matmul path**, so the
scaler tile is the second matmul operand and `Σ = Σ_j input[j]*scaler[j]`; zeroing
`scaler[j]` drops column `j` regardless of its contents.
- Reader emits **two** scaler tiles (`prepare_partial_reduce_scalers<cb, SUM, REDUCE_ROW,
  partial_w>`): tile 0 = full `1/W`; tile 1 = `1/W` in `[0, partial_w)`, zeroed in
  `[partial_w, 32)`. (Scaler CB sized to 2 pages.)
- Compute selects with `ReducePartialScaler::last_tile_at(1)` on the global last W-tile
  (`c == num_chunks-1`), `none()` elsewhere.
- Input-garbage-agnostic → the `test_rms_norm_w_padding_is_masked` proof (padding stuffed
  with 99.0) passes because `99.0 * 0 = 0`. This is why W (the reduce axis) is masked in the
  scaler while H (not reduced) needs only the correct tile-row count, no masking.

### 6c. Skill/header redundancy (softened process note) 🟡
The implementation matches the `partial-scaler-reduce` SKILL's recipe essentially
line-for-line (2-tile CB, `prepare_partial_reduce_scalers`, the `last_tile_at(1)` ternary),
but the breadcrumbs show that skill was **never invoked** (only `numeric-formats-metal` and
`memory-layouts` were). This is **not** a competence gap: the same recipe is fully
documented in the helper headers (`reduce_helpers_dataflow.hpp` /
`reduce_helpers_compute.hpp` docstrings), and "explore the helper library" is the
implementer's standing discipline — so the method was recoverable (and recovered) without
the skill. The mild observation is **skill/header redundancy**: this skill mirrors the API
docs rather than adding non-recoverable knowledge (anti-patterns, "looks-right-but-hangs"
warnings), so skipping it cost nothing here. Skills that only restate header docstrings add
little; skills that encode failure modes not in the headers are the ones whose non-invocation
would actually matter.

### 6d. Scope deferrals (clean) ✅
- Regime B gated to aligned-W only (`partial_w == 0 && R == 1 && NC >= 2`); non-aligned
  wide-W falls back to row-parallel Regime A. Deferred corner, later handled in R5b.
- `bf8b + non_tile_aligned` needed **no** EXCLUSION (the R1 verifier had flagged it as a
  *possible* refusal): `cb_xsq` is fp32 and the partial scaler masks the columns, so bf8b
  non-aligned just works.

---

## Finding 7 — Refinement 3 (ROW_MAJOR / tilize-wrapped): two low-severity, self-correcting footguns ✅

**Severity: low.** Both defects were caught by a mechanical safety net and any competent
agent would notice and fix them. The cost was **a few wasted turns / tokens, not a
correctness risk** — they never had a path to shipping silently. Filed for the *pattern*,
not the danger.

R3 added ROW_MAJOR to `SUPPORTED["layout"]` + `["gamma_layout"]` via an in-kernel
tilize-wrapped Regime A (reader streams RM sticks → compute tilizes prologue → existing TILE
math unchanged → pass-2 untilizes epilogue; `CHUNK = largest divisor of Wt ≤ 4` for a
uniform constexpr tilize width). No host `to_layout` (honors the `memory-layouts` native
policy, which the implementer *did* invoke, alongside `Explore` for the tilize API). No
hangs, no expert-debugger escalation, clean landing (golden RM INTERLEAVED 980/0/0).

### 7a. Chained lossless-tilize precondition — caught at COMPILE TIME ✅
Keeping fp32 lossless through the RM path needs the tag on **every** CB in the chain, not
just the square's input. `Fp32Mode::Lossless` `static_assert`s that its tilize-input CB is
`UnpackToDestFp32`-tagged; the implementer had tagged `cb_input_sq` (from 1b) but not the
new `cb_rm_in_sq`. Compile-time `static_assert` → quick fix (`5d0d850415`). The *good* kind
of footgun: impossible to ship silently.

### 7b. Regression in an already-supported dtype via an unguarded shared path ✅
The RM byte-geometry computed `input_tensor.element_size()` **unconditionally**, but
`bfloat8_b` (block format) has no valid `element_size()` — so adding RM support **regressed
bf8b TILE cells** that had passed since R1. Root pattern: *a new axis value's code ran on a
path shared with all existing values and computed a quantity invalid for one of them.* Fix:
guard the byte-geometry behind `act_is_rm`/`gamma_is_rm` (`75c96a41f1`). **Caught by the TILE
non-regression suite, not by any RM test** — the whole reason non-regression suites exist.
The one real lesson: an agent that runs only the new-axis tests (not the full non-regression
sweep) would ship this. The safety net worked *because it was run*.

### Severity contrast (the useful axis)
7a/7b sit at the opposite end of the spectrum from Finding 1: **loud + mechanically caught +
self-correcting** vs **silent + PCC-invisible + needs an expert-debugger and an all-ones
repro**. Same op, very different risk. For evaluating the pipeline, the metric that matters
is not "did it hit a bug" but "did a mechanical gate catch it, or did it need deep
semantic debugging?" 7a/7b cost turns; Finding 1 cost correctness confidence. R3 also shows
the implementer did **not** repeat the R1 CB-sharing mistake (new RM CBs are clean SPSC
handoffs; Regime B sharing correctly deferred to 3b, reusing the fixed `cb_combine` route).

---

## Finding 9 — Test-coverage gap: roundish shapes, no adversarial shard geometry, and a self-graded bug class ✅

The most important finding for evaluating the pipeline, because it is about what the
pipeline **structurally cannot see**, not a bug it happened to hit.

### Evidence
- **Unit tests (implementer-authored)** use mild, roundish shapes. R4's
  `test_rms_norm_height_sharded.py` `_SHAPES`: 6 tile-aligned (all powers of two:
  32/64/128/256/512/4096) + exactly **2 non-aligned** (`W=50`, `H=47`), both mild.
- **Golden suite (`feature_spec.py INPUTS`)** is broader — `W ∈ {50,17,47,100}`,
  `H ∈ {17,50,47}`, both-non-aligned, rank `[2,3,4]`, wide-W to 32768 — but still roundish:
  * **No extreme partial fractions.** No `partial_w = 1` anywhere (`W=33`, `W=65`) — the
    nastiest masking boundary (1 valid column, 31 zeroed) is untested. No `W=31`. The
    `alignment` axis is 3 coarse buckets (`tile_aligned / w_non_aligned / h_non_aligned`)
    and **cannot distinguish `partial_w=1` from `partial_w=17`**, so nothing forces the
    extremes; the suite samples a couple of mild representatives per bucket.
  * No prime/awkward dims (batches are `1,2,4,8,32`).
- **Sharding geometry is structurally outside the grader:**
  * The sharded golden `INPUTS` are **all tile-aligned, roundish** (`(1,1,256,512)`,
    `(1,1,32,2048)`). No adversarial shard geometry at all.
  * **Shard spec is not a declared axis.** `memory_layout` is an axis, but the shard
    *shape* (grid, per-core rows/cols, even vs. remainder split) is **derived** by
    `auto_shard_config`, which produces **canonical** splits, not adversarial ones. The
    registry cannot enumerate shard geometries, so the golden cartesian can't cover them.

### The load-bearing conclusion
**Every sharding bug in this run lived in that ungraded space** — the 4b `37×7 > 256`
over-cover hang, the 5c `shard_w=8` neighbor-gamma read (PCC 0.52), the byte-vs-tile
gamma OOB — and **none would be triggered by the golden suite's aligned roundish sharded
INPUTS.** They were caught only by the implementer's **own** debugging probes
(`probe_024/025/028`, all-ones, sub-tile `[5,8]`). So a whole class of sub-tile /
uneven-shard bugs is caught (if at all) by implementer diligence, **not by the grader**. Had
the implementer not happened to probe `shard_w=8` or hit a 7-core split, those bugs would be
**latent AND ungraded** — the suite shows green.

### Calibration (don't over-fear the shape half)
The W-masking is a scaler multiply-by-zero that zeros `[partial_w, 32)` regardless of
`partial_w`, so `partial_w=1/17/31` all hit the same path — a value-specific masking bug is
*less* likely than the roundness suggests (`W=17` already covers "last tile entirely
partial"). Shape-roundness is a **moderate** risk. **Sharding geometry is the serious one**:
a genuinely different code path per geometry, and the pipeline's structural blind spot.

### Why the pipeline is weak here (both fixable)
1. `auto_shard_config` generates canonical splits; there is **no adversarial-shard
   generator** that deliberately produces tiny-remainder / sub-tile-in-one-or-both-dims /
   prime-core-count / uneven-per-core shards.
2. Shard geometry isn't an axis, so `SUPPORTED`/`EXCLUSIONS` and the golden cartesian
   can't reason about it; coverage is only whatever hand-authored `_SHARDED` cases someone
   thought to write — and they wrote aligned roundish ones.

---

## Proposal — an adversarial latent-bug / edge-case agent

Direction under consideration (user): a dedicated **adversarial tester** agent that hunts the
ungraded space above and generates tests for it. Sketch of what would make it effective:

**Mandate.** Not "does the op pass its tests" (verifier's job) and not structural read-only
review (ttnn-static-analyzer's job) — instead a **dynamic edge-case fuzzer**: synthesize
inputs/geometries the declared axes can't express, run them, and surface value/scale/hang
failures as latent-bug findings + regression tests.

**Independence (critical — avoid inheriting the implementer's blind spots).**
- Its own fp32 reference, never the implementer's `_ref`.
- **Scale-sensitive metrics by default**: relative-RMS *and* the got/true **ratio
  distribution**, not PCC-only. (PCC-only is exactly what hid Finding 1.)
- Its own tolerances tied to the golden bands, not chosen ad hoc.

**Target space (the ungraded corners).**
- *Shape*: extreme partial fractions (`partial_w ∈ {1, 31}` → `W=33,63,65`), single-partial-
  tile (`W<32`), prime H/W, tiny/prime batches, `H`/`W = 1`.
- *Shard geometry* (highest value): tiny-remainder shards, sub-tile shards in one **and**
  both dims, prime/awkward core counts, uneven per-core splits, grid-boundary splits — i.e.
  an **adversarial-shard generator** to sit beside `auto_shard_config`.
- *Determinism*: all-ones, monotonic, and per-position-marker inputs to expose scale/alias/
  neighbor-read bugs that random+PCC hides (proactively, not reactively as the implementer
  did).
- *Timing*: run under `--dev` (accessor/LLK asserts) and both dispatch grids (the 56 vs 64
  core disparity that flipped regime selection in Finding 1/3).

**Pipeline placement & guardrails.**
- Runs after the verifier, as a gate before a refinement is marked `[x]`.
- **Must not edit `eval/golden_tests/`** (owned by the user; see 5d). It writes adversarial
  *unit* tests and reports findings; genuine gaps become INVALID/EXCLUSION escalations or
  refinement candidates.
- Must distinguish real failures from (a) legitimate EXCLUSION refusals and (b) the
  L1-fragmentation artifact (re-run failing cell in a fresh device) — else it drowns in
  false positives.

**Expected catch on this very op:** the 4b over-cover, the 5c sub-tile gamma bugs, and any
`partial_w=1` masking boundary — all of which the current grader misses.

---

## Finding 10 — Missed optimization: `cb_input_sq` could be a zero-copy CB alias, not a copy 🟡

For fp32 **TILE / interleaved / sharded** input, the op fills a dedicated `cb_input_sq`
(`UnpackToDestFp32`-tagged) with a **copy** of the resident/read input — a local L1→L1 NoC
copy in the sharded case, plus a `CHUNK`-page L1 allocation. This is avoidable.

**The mechanism the pipeline left on the table.** `CBDescriptor` has **one** backing
allocation (`buffer` / `tensor`) but a **vector** of `CBFormatDescriptor`s, each with its own
`buffer_index`, `data_format`, `page_size` (`tt_metal/api/tt-metalium/program_descriptors.hpp`).
Two format descriptors — `buffer_index=CB_INPUT` (default) and `buffer_index=CB_INPUT_SQ`
(fp32) — over the **same** allocation make both CB ids alias the same L1 bytes. The
`UnpackToDestFp32` tag is applied *separately* on the ComputeConfigDescriptor's
`UnpackToDestModes[CB_INPUT_SQ]` (the format descriptor has no unpack field), so it composes
cleanly. Then the square reads `CB_INPUT_SQ` (fp32→DEST lossless) and pass-2 reads
`CB_INPUT` (FPU/TF32) from the **same bytes** — no copy, no extra CB.

**Caveats (why it's 🟡, not a slam-dunk):**
- **Sharded-tensor helper is single-format.** `cb_descriptor_from_sharded_tensor(cb_id,
  tensor)` takes one `cb_id`; every in-tree use is one-format-per-call and there is **no
  tensor-backed multi-format example** in the repo. The struct permits `tensor=` + two
  formats (append a `CBFormatDescriptor` to the returned descriptor, or build it via the
  normal `ttnn.CBDescriptor(format_descriptors=[fmt0, fmt1], tensor=...)` create), but that
  combination is untrodden — needs an `Explore`/probe to confirm the CB-sync/pointer
  bookkeeping honors two `buffer_index`es over one resident buffer (SPSC-adjacent).
- **RM can't use it.** For ROW_MAJOR, `cb_input` is a TF32-truncated *tilize output* — the
  full-fp32 `x` isn't resident, so there's nothing to alias; RM genuinely needs the
  lossless-tilize into a separate fp32 buffer. Adopting aliasing therefore reintroduces a
  layout branch (alias for TILE/sharded, real copy for RM) and loses the single uniform
  `unary<Square<>, cb_input_sq, cb_xsq>`.

**Assessment:** a legitimate L1/NoC saving on the fp32 TILE/interleaved/sharded path, almost
certainly traded away for **code uniformity with the RM path** rather than because it was
impossible. Low urgency (perf/L1 only, correctness fine), but a clean example of the
implementer optimizing for one-code-path simplicity over a layout-specific win.

---

## Finding 11 — RM data movement: direct-tilize vs gather, and the bounded-L1 tension ✅ (architectural note)

Not a bug — the discriminator for *why* the RM path costs what it costs, and where a
zero-copy/direct-tilize optimization is (and isn't) possible. Complements Finding 10.

### The fundamental RM tax
The math is tile-based (32×32); RM stores row-major sticks. So RM *always* pays a **layout
conversion** TILE never does: `tilize` in, `untilize` out. That conversion is **compute**
(TRISC unpack→DEST→pack), not NoC — but *feeding* it costs NoC (`read_sticks_for_tilize`).

### Direct-tilize discriminator (can you skip the gather?)
The `tilize` LLK consumes a CB as **contiguous tile-row blocks**: 32 rows tall, physical row
stride == tile-block width, width padded to a 32 multiple, full tiles. If resident data
already matches that, you can bind + tilize in place. `read_sticks_for_tilize` exists for
when it doesn't — its own docstring: *"pads the L1 stride"* (non-aligned W), *"pushes full
tile pages for the partial last"* (non-aligned H), and `byte_offset_within_page` to *"select
a chunk along W"* (column slice).

- **CAN tilize directly:** full-width, `W%32==0`, contiguous rows already resident in L1 —
  e.g. a small HEIGHT-sharded RM shard with full-W tile-aligned rows.
- **CANNOT (must gather/pad):**
  1. **non-aligned W** → rows aren't 32-padded; stride is `W` not `ceil(W/32)*32`.
  2. **column slices** — WIDTH/BLOCK sharding *or* W-chunking → source row stride ≠ slice
     width; needs the `byte_offset` gather (sub-tile `shard_w` also hits the sub-32B-offset
     alignment problem, cf. 5c gamma bug).
  3. **non-aligned H / sub-tile shards** → partial last tile-row; direct 32-row read goes
     OOB (literally the 4b `37×7>256` hang).
  4. **DRAM source** → can't tilize from DRAM; must NoC-read into L1 first anyway.

### The bounded-L1 vs direct-tilize tension (the forcing function for this op)
A *direct* tilize needs the **entire W-wide row resident** in `cb_rm_in` → L1 grows with Wt.
The design pins `cb_rm_in` to `CHUNK` tiles (must NOT grow with Wt, design rule P3), achieved
by reading W in `CHUNK`-wide **column slices** — each a `byte_offset` gather. So:

> **You cannot both keep `cb_rm_in` bounded (chunk along W) AND tilize the full row directly.**
> For wide W the memory-boundedness requirement forces the column-chunked gather regardless
> of alignment.

Consequence: rms_norm uses `read_sticks_for_tilize` on **every** RM path — even aligned
full-W where the shape *alone* would permit a direct tilize — for (a) bounded L1 on wide W
and (b) one uniform reader/compute across aligned/non-aligned/sharded. So the "direct tilize
is possible" case (small, aligned, full-width, resident) is real but deliberately unused
here; the bounded-L1 discipline + uniformity push everything through the gather.

### Net for the tracker
RM is structurally heavier than TILE (layout conversion), and for **sharded** RM the gap vs
TILE is specifically that **TILE storage == compute layout → zero-copy bind**, while **RM
storage ≠ compute layout → gather + tilize** (Finding 8/10). The only genuinely-sheddable RM
NoC cost is the **fp32 duplicate stick gather** (`cb_rm_in` + `cb_rm_in_sq` hold byte-identical
sticks → aliasable, Finding 10 one layer up), with the tilize-pops-its-input sync caveat.

---

## Cross-cutting theme

Several findings share one shape: **a correctness rule exists in the codebase's prompt
surface, but not at the pipeline stage where the decision that needs it is made.** SPSC
lives with the debugger but the CB is born at the planner. Scale-vs-precision triage lives
in the debug skill but the call is made at verify time. Regime-pinning discipline emerged
in refinement tests but the branching decision is a planner artifact. The highest-leverage
fixes push each rule *upstream* to the stage that first commits to the design choice it
governs.
