# Seeded generation for op parity — working checkpoint

*Started 2026-09-02. Written so this thread can be resumed cold. §1 is op-agnostic and is the
part that matters; everything from §5 on is playground evidence for one op (`rms_norm`) and is
illustrative, not normative.*

---

## 1. Assumptions and requirements

These govern the mechanism, not any one op. Numbered so later sections can point at them.

### A. Scope

1. The generated op **replaces the op it targets** in the tree — not a demonstration of parity.
2. The mechanism **generalises to ops with no counterpart to target**. The delta may come from
   any source: a comparison against an existing implementation, a human wishlist, a model
   requirement, a bug report.
3. Changing the op's **implementation mechanism** (host language, program-construction API) is
   out of scope for a gap-closing generation. That is a separate decision.
4. The successor is generated under a **staging name**. Adopting the target's name is a distinct
   cutover step.

### B. Requirements on the generated op

5. **No regression on anything the seed supported** — correctness and per-configuration
   performance.
6. **When an optional input is absent, the compiled program is equivalent to the seed's** for
   that configuration: same buffers, same code path, same blocking.
7. **Nothing is allocated for an absent optional input.**
8. **Nothing is *sized* for an absent optional input.** Blocking and buffer sizes are computed
   from the actual call configuration, never from a worst-case bucket over all configurations.
9. **Optionality is expressed by compile-time specialisation**, never by materialising an
   identity value (a zero tensor, a ones vector).
10. **The op's dispatch contract is preserved** — added capability must not turn one device
    dispatch into a sequence of them.
11. **Any deviation from 6–8 is justified by measurement**, not by argument.
12. **Where the seed outperformed the op being replaced, that advantage is preserved** — but the
    regression gate is against the **seed**, not against the target.

### C. Process

13. **Integrity holds**: the pipeline reads only explicitly designated seeds, never an
    implementation it is meant to replace.
14. **Designation is declarative** — stated in the op's prompt. Not inferred from path,
    authorship, or commit time.
15. **One-shot generation**: the planner sees the full input set at once, so the blocking design
    accounts for it rather than retrofitting it one operand at a time.
16. **Preserve and extend, not redesign.** The seed is the starting *code*, not a blueprint.
    Gratuitous divergence is the failure mode here — not copying.
17. **Ground truth comes from the seed's code**, not from its planning artifacts. Those are
    write-once and go stale (see §6).
18. **A human ratifies the delta** before a run. It is the entire specification for that
    generation.

### D. Verification

19. **Requirements 6–8 are mechanically checked**, not asserted — build the program with and
    without each optional input and compare the resulting footprint.
20. **The regression gate is mechanical** — the successor's test suite extends the seed's, so
    shared cases keep their identity across generations.

---

## 2. What we are trying to accomplish

Make an AI-generated op a drop-in replacement for an existing hand-written one, and make that a
**repeatable capability** rather than a one-off.

Two possible framings; we chose the second:

- *"Be compatible with an existing symbol"* — a migration tool, works only where a counterpart
  exists.
- **"An op grows capabilities over time"** — each generation is seeded by the previous one plus a
  delta spec. Parity with an existing implementation is one consumer of that machinery
  (requirement 2).

### The approach

1. Generate the op from scratch (already done for the playground op).
2. Produce a **gap spec** — what the generated op lacks relative to the target.
3. Generate the successor **seeded by (1) plus (2)**, in one shot.

Integrity is preserved (13): the pipeline reads its *own* prior output. Only the gap-analysis
skill reads the target, and it runs outside the pipeline — the same position
`/translate-reference-tests` already occupies.

### Why one shot rather than incremental refinements

New optional operands consume the memory budget that decides an op's blocking regime. Added one
at a time to an existing op, each retrofits into a design that never anticipated it, and each
addition can demote regimes. A seeded regeneration lets the planner design the blocking with the
full input set known up front (15). That is the whole argument for the approach.

Note this does **not** conflict with 16: one-shot describes *when the planner sees the inputs*,
not how much it rewrites. The expected output is largely the seed plus extensions at the seams.

---

## 3. State — what exists

*Built and run 2026-09-02/04.*

### Built

| Component | State |
|---|---|
| **`/analyze-op`** | written, run once, twelve first-run defects fixed |
| **`/analyze-native-gap`** | written, run once, twelve first-run defects fixed |
| **`/golden-tests` Mode C** | added — extend a seed's prompt and suite into a successor; carries the inconsistencies into the successor's prompt |
| **Seed designation** | F1 closed. `prompts/seed.txt` + a `{seed_context}` slot in planner/implementer/verifier/refinement; integrity defers to it; declared by a `# seed: <path> [<branch>]` header. Submodule `c86abd4` |
| **Perf-case widening** | 13 → 19 cases; the block-sharded reference restated; `/analyze-native-gap` gained placement/extent stratification (G19). Submodule `4d5dbe1` |
| **Provenance script** | `eval/provenance/rms_norm_ttnn_baseline_measurement.py` — all 19 targets re-derivable. Cannot run where it lives; its banner says where. Submodule `ba0b374` |
| **Precision convention** | the fp32 + 16-bit-accumulator cell is supported and scored, not an `EXCLUSIONS` entry. It is frequently what an op's own default compute config produces, so the old rule made an op refuse its own default. Submodule `2ed265d` |
| **Counterpart agnosticism** | 20 leaks removed from prompt, both specs, two skill descriptions; mode C no longer regenerates the framing. Submodule `15c857a`, `b4294a3` (§4c) |

Both analysis skills gained steps after their first runs: the gap skill now **mandates a usage sweep** (parallel subagents, one per parameter or file group) and an **independent fidelity audit** by a reader that did not produce the manifest, scoped to the items that required judgement — matches and pure renames are skipped, `--audit-exhaustive` overrides. Gates G16–G18 enforce all three.

### Produced for the playground op

| Artifact | Where | State |
|---|---|---|
| Description of the seed as shipped | `ttnn/ttnn/operations/rms_norm/op_as_shipped.{json,md}` | audited by two independent readers, corrected, all gates pass |
| Difference list vs the target | `.claude/eval/golden_tests/rms_norm/native_gap.{json,md}` | 28 items, gates pass, every item settled |
| Successor prompt | `.claude/eval/prompts/rms_norm_ttnn.txt` | 446 lines, regenerated after the inconsistencies landed; carries them as `## Inherited interface quirks` and designates the seed's branch alongside its path |
| Successor suite | `.claude/eval/golden_tests/rms_norm_ttnn/` | eight files incl. a stand-in for the target's tuning-object type; every seed case survives with matching identity. Regenerating after the prompt-only changes produced byte-identical content, verified by full seed diff. |

### What the audits found

The description was sound in substance and unreliable in bookkeeping: **159 of 212 source references pointed at the wrong line**, while the numbers themselves were confirmed exactly by device probes across 23 of 25 cases. Three semantic errors, four boundary errors, seven omissions — all corrected. Two new gates followed: every reference must be read back and confirmed, and keyword-only arguments must be recorded.

### What the target's interface claims but does not honour

The gap manifest carries a `## Native op inconsistencies` section — **14 entries across 10 classes**, produced by the mandated usage sweep. The discriminator is always *how a parameter is used*, never what the signature says: a field with a hundred call sites all passing the same derived value is not a degree of freedom.

Classes: accepted and never read · validated then discarded · exposed but pinned by an assertion · fails as a raw crash · documented but unenforced · documented but contradicted · reference function ignores its own inputs · accepted but wrong · entry point narrows what its own implementation supports · assertion that cannot fire.

The sweep also **refuted a belief we held**: only one compute-settings field is dead. `math_approx_mode` is live through a compile-time constant, and the trap is that the kernels carry an explicit template argument at that position which is a *different* knob — so grepping the kernels reads as false confirmation.

Findings worth carrying: the output-placement argument is accepted and discarded whenever write-over-input is set, and a test passes both in one call; a host-storage input dereferences a null device instead of hitting its documented check; a documented, guard-clean precision combination returns near-zero correlation, with an in-tree sweep coercing types to dodge it; the rank-0 path drops the residual operand as well as the epsilon term; and the one sweep case covering bias and rank-0 has **two assertions that cannot fail** — which is what makes every "tested: 0" a measured zero rather than an absence of evidence.

### The carry-through corrected a wrong instruction

The successor's prompt, before the inconsistencies existed, told an implementer *"the blocking fields are the CALLER'S choice and MUST be honoured."* That is false: the block dimensions are pinned by assertion to restate the input's shard geometry, the core grid is accepted and never read, and the sub-block width is the only free knob. The regenerated prompt says so, and adds five refusals and two obligations that were not there — including that a precision combination the target accepts and computes wrongly must be **correct or refused**, never accept-and-garbage.

Had the prompt been hand-patched instead of regenerated, that instruction would have survived into a generation run. This is the argument for running the mechanism rather than editing its output.

### Decisions taken

| Question | Decision |
|---|---|
| Rank 0 | Supported, computed correctly with the floor term so a zero input yields zero, inside one device program. The target's own path omits the floor and chains four operations; neither is reproduced. |
| Block-compressed scale tensor at tiled layout | Not a difference — both sides accept it. The same format at row-major is physically impossible, so a guard against it is redundant on either side. |
| The target's reference function | Write a correct one. Theirs discards two operands and cannot detect a fault in either. |
| The two entries about how the target sizes its buffers | Withdrawn — they describe its implementation, not a difference in what the two ops accept. Records kept. |

### What the target's tuning object actually permits

Investigated in full; the summary matters because it dissolves an apparent conflict.

- **The caller's blocking is not authoritative — the shard is.** Exact-equality assertions force the block dimensions to restate the input's shard shape. All **100 construction sites across 60 files** do exactly that.
- **The core grid is validated then discarded** — checked against the shard, extracted into a local, never read. Five sites pass a differing grid; all legal, all inert.
- **Three fields carry real meaning**: the sub-block width (the only free blocking knob, and the one callers tune — 83 of 100 pass something other than the default), write-over-input, and the algorithm switches.
- **For an unsharded input the caller has no influence at all.**
- **Three mismatches are reachable, untested, and two fail as raw C++ exceptions** rather than diagnosable errors.

So choosing our own blocking from a memory budget conflicts with nothing anyone exercises.

## 4. What remains

**Test count — DECIDED 2026-09-04: leave it at 3.6×.** The successor's suite is 736 axis-combinations against the seed's 204, and the growth is *entirely* `gamma_mode` — 164 combinations per operand-carrying value against 40 for one without, so four operand-carrying values are the whole multiplier. Rank grew 3→6 values for free (shape-derived, does not multiply).

Reductions were priced and rejected. Dropping presence values costs 2.0–2.8× and removes whole compiled paths from the sweep. Pinning the per-channel operand's format on the three new presence values reaches 1.61×, and was the leading candidate until the evidence came in against it:

- **Upstream exercises that exact cell deliberately.** A named mix-precision table alternates the operand dtype specifically *"so that both formats appear in every {LN, RMSN} × {G, GB} category"* — i.e. weight-dtype variation *inside* the with-bias category was the point. For rms_norm, **6 of 8 mixed cells carry a bias, a residual, or both**, across three consumers.
- **Our own suite targets it nowhere** — no loose case, no regression test, and `TOLERANCE_OVERRIDES` is keyed only on `(dtype, fp32_dest_acc_en)`. It exists purely as cartesian fallout.
- **The skill sanctions no such collapse.** Only two are allowed: share one axis across co-present weight tensors (already done) and canonicalise the absent-weight cell (already done). Its interview calls the linked-dtype option *"blind to mixed-precision"*.
- **Proportion.** 26,206 device tests is *smaller* than run 893's 40,722, which ran through twelve phases. The multiplier alarms; the absolute is under precedent, and the absolute is what costs wall-clock.

Two mechanical facts found while pricing this, both worth keeping. `INVALID` is the only tool that narrows the cartesian, and it **also suppresses hand-written cases at the same cells** — so "pin the sweep, add the corners back as loose cases" is self-defeating under the current harness. Separating *declared* from *swept* would fix that and the residual-out-of-sweep question at once; it is infrastructure, not a spec edit. And a pin on all four new presence values silently **deletes** the residual-only mode, whose canonical cell is the `("none","none")` sentinel.

**A correctness question surfaced by accident, and was settled 2026-09-04 by measurement.** Two upstream sources disagreed on whether an fp32 weight with a bf16 input works: a model-traced sweep hard-codes the weight dtype to the input's, commenting that a float32 gamma on a bf16 input *"produces garbage (≈0 PCC)"*, while a nightly test runs bf16 input + fp32 gamma + fp32 bias and asserts **PCC ≥ 0.999**.

Measured on a built main tree across a 64-cell grid (2 shapes × 2 input dtypes × 2 weight dtypes × 2 weight layouts × 2 placements × bias present/absent): **every cell passes, minimum PCC 0.999984**, and the mixed cells are indistinguishable from matched-dtype ones.

Both sources were right, about different cells, at different times. The comment landed 2026-06-29; the bug was fixed 2026-07-18, nineteen days later, by a PR whose own table records fp32 ROW_MAJOR gamma going **0.314757 → 0.999975**. The reader selected whenever gamma is ROW_MAJOR hardcoded bf16 byte counts, so an fp32 gamma fetched only channels 0–15 of every 32-channel tile and read uninitialised L1 for 16–31. It keyed on gamma **layout**, not dtype — the sweep builds a ROW_MAJOR gamma and hit it; the nightly builds a TILE gamma and never did — and it was interleaved-only, the sharded writer having had the fp32 branch since 2024.

Consequences: the sweep's comment is stale and its `w_dtype = input_a_dtype` default now **suppresses mixed-dtype coverage the op supports** (an upstream follow-up, not ours). And this closes the loop on the test-count decision — the `gamma_dtype` axis defends a combination that works and that had a real bug within the last two months, so pinning it would have dropped coverage of a freshly-repaired path.

**Not started:****Not started:**

| # | Item | Note |
|---|---|---|
| F2 | Planner reads the prompt's rules section explicitly | It sees them only implicitly today. Note the planner *does* already receive the whole op prompt — `{requirements}` is the prompt body verbatim — so this is about emphasis, not delivery |
| F3 | ~~Preserve-and-extend wording~~ | **CLOSED 2026-09-04.** It was never the balancing act it was billed as — requirements 6 and 16 partition rather than conflict. See below |
| — | Per-item usage records | Independently re-confirmed 2026-09-04: **0 of 28 items carry a `usage` record**, so G16 fails for every parameter item. The sweep ran but merged only into `inconsistencies`; `native_gap.md` has no `## 6. Usage evidence` section |
| — | Independent check of the successor's prompt and suite | The fresh-eyes pass, agreed but not run |
| — | Ownership of the tuning object's type | The suite now has a stand-in; who defines the real one is undecided |
| F4/F5 | Regression gate and footprint-equivalence check | Can be manual for the first run |
| P1 | Reference-test discovery by symbol, not filename | Deferrable for a playground run |

**Then:** the first seeded generation run.

**As of 2026-09-04 the only item gating a start is the fresh-eyes pass.** F1 and F3 are closed, the test-count and perf-focus questions are decided, the counterpart scrub is done and mode C no longer regenerates it, and the blind layer is built (§4d). The prompt and the suite have both changed substantially in two days — the quirks section rewritten, six perf cases added, a reference restated, a new rule — and nobody independent has read either since.

### F1 — closed 2026-09-04

The integrity text forbade reading any pre-existing implementation of the op *"whether in the working tree or in git history"*, which caught the designated seed. Two collisions, not the one recorded: the categorical bullet, **and** the history bullet, which the successor prompt violates by designating the seed's branch.

Prohibition and permission now reference each other. `integrity.txt` keeps the prohibition and names its one exception; `prompts/seed.txt` holds the exception — what is readable, what stays out of bounds, and the three obligations (preserve-and-extend, do not regress, code-not-documents is ground truth). Rendered into a `{seed_context}` slot for **planner, implementer, verifier and refinements** — all four run under the same integrity text, so an agent given the prohibition without its exception refuses the seed mid-run.

Two details worth keeping:

- **The branch carve-out is path-scoped.** A run branch nukes the native op, so the nuke commit's *diff* contains the whole target. Only commits touching the designated path are readable.
- **Declaration is a `# seed:` header**, beside the existing `# golden:` line — so seeding is greppable and auditable rather than buried in 446 lines of prose. `test_prompt_templates` pins the slot the way it pins `{integrity}`: a template carrying one without the other is the exact silent failure F1 was.

## 4b. Performance — what we learned, and what is now filed

Two days of work here started from "add a few perf cases" and ended with three issues, because the perf machinery turned out to be held together by convention rather than mechanism.

### The tournament, as it actually works

Measure per-stage first, rank by measured headroom, gate against a roofline; pick a focus shape; float a portfolio of ideas; fan out one subagent per idea in parallel; aggregate the fastest non-conflicting set; **spread each winner as the op's one unqualified path everywhere and delete what it replaced.** A guard is earned only by a correctness impossibility or a *measured* material regression — untested is not an exception, flat is not an exception. So the guard set is the only thing that can narrow a spread, and any regime it omits is a hazard by construction.

### What run 893 revealed

Run 893 (`dnijemcevic/2026_08_04_1241_rms_norm`, tip `bfe3e673a6a`) ran three tournament rounds, ~$373.

- **All three rounds free-selected the same shape**, `(1,1,8192,1024)` block-sharded, and recorded *verbatim, three times*: "`feature_spec.py` carries no `attention:` note, so the focus shape was free-selected." The flag the perf agents look for exists on exactly one suite in the tree, and it is not this one.
- **Free-selection still landed correctly** — it ranked the 13 by measured ÷ their own reference and took the worst. That procedure is written down nowhere but the changelog it produced. Ranked by absolute time instead, three rounds would have gone to a shape already 1.6× *faster* than its reference.
- **Five shapes regressed and nobody saw them.** 5,436 tests carry a device time in every phase; diffing the last generality refinement against the final perf round shows five materially slower (>10% and >2 µs), the worst 184,790 → 821,128 ns. None is confirmed — the point is that they were never surfaced, so never judged. All five sit outside the corner the guard set covered.
- The DB's **run-level rows for 893 predate all three tournaments** — they match Refinement 4, so anything reading the un-phased results sees the op 3.4× slower than it finished.

### The references were real, and one was wrong

Re-derived all 13 against native `ttnn.rms_norm` on a pristine main clone (`/localdev/dnijemcevic/tt-metal2`, built with the profiler). **12 of 13 reproduce within ~3%**; AICLK read from the profiler preamble was exactly the reference 1350 MHz.

The exception: `(1,1,8192,1024)` block-sharded, recorded 25,640, measured 28,619 (+11.6%). Root cause is the program config — native's auto-derived `subblock_w=1` cannot reach it; `subblock_w=4, inplace=True` gets 25,513. The record was achievable but under-specified, so it is **restated to 28,619**, the tuned figure recorded beside it. Note 893's generated op finished at 24,307 — under both.

### Hard constraints found while measuring

- **A sharded residual costs a second full resident shard.** Native sizes the residual buffer to the input's (`in1_dfb_size = in0_dfb_size`) and never streams it. It fits to **120 tiles per core** and overshoots L1 by 93 KB at 128. Bisected to the byte: each 4-tile step closes the gap by 47,104 B, and 120 fits with 1,024 B to spare.
- **`fp32_dest_acc_en=True` does not fit block-sharded at all** at that extent, not even weight-only — the static buffer region grows ~1.8×. A block-sharded case carries the residual **or** the accumulator, never both.
- **A sharded residual is mandatory, not optional.** Two `TT_FATAL`s require the residual to be sharded with an identical shard spec whenever the input is; sharded-input + interleaved-residual is unreachable. Models pass sharded residuals deliberately — `bge_m3` engineered away the reshard for a measured −1.1 µs/call. What is untested is a sharded residual *at large extents*: upstream tops out at ~32 tiles/core.
- **Streaming the residual is where the successor could beat native**, which has no streaming option at all.

### Production does not look like our perf cases

PR #127 (`dnijemcevic/translate_traced_tests`) snapshots 388 recorded `ttnn.rms_norm` calls → 140 distinct configs, **334,159 executions** across Qwen, Llama, Gemma, Mistral, Whisper, SDXL.

| | traced production | the original 13 |
|---|---|---|
| placement | 110 interleaved, 30 width-sharded; **zero block-sharded** | includes a block-sharded case |
| `fp32_dest_acc_en` | **True** in 110 of 140 | pinned False on all 13 |
| weight layout | **ROW_MAJOR, DRAM** — constant across every trace | TILE |
| `bias` / `residual` | **never passed** | – |

And the row-major weight is **faster**: 65,349 vs 76,616 ns on an identical config (0.853×). So the existing references are loose by ~17% for the weight layout production actually uses.

Caveats on the trace, which may itself be the flawed input: one snapshot day, 24-hour export window, symbol-scoped (fused/CCL paths invisible), compute config untraced in 21 of 140, and a circular signal — a trace cannot distinguish latent demand from absent demand.

### What changed, and what is filed

**Changed** (submodule `4d5dbe1`): perf cases 13 → 19, adding the operands at `fp32_dest_acc_en=True` (each paired with a weight-only baseline at the same setting, so the operand multiplier is never confounded with a precision change), the block-sharded residual at 112 tiles/core with ~95 KB headroom, and the ROW_MAJOR-weight case taken verbatim from the most-executed traced config. `/analyze-native-gap` gained `coverage.strata` and gate **G19**: a bare scalar case count on a tensor-shaped item can distinguish neither *covered everywhere* from *covered only interleaved*, nor *covered* from *covered only small*. `baseline_measurement.py` lands beside the successor's spec — it cannot run there (the native op is nuked) and its banner says so.

**Filed, for the team:**

| Issue | Question |
|---|---|
| **#175** | A perf model for achievable targets. No written procedure exists for obtaining an `achievable_ns`; the numbers were hand-measured twice (the first set against a wrong core count, silently wrong for a day) with no script. Proposes a computed floor — data movement **and** compute **and** dispatch overhead, taken as a max — because a DM-only floor is 61–100× off on decode and divides by ~zero on resident shards |
| **#176** | How a guard set should be built. Agent-chosen today from a prose predicate; run 893's covered one cell of the precision × layout × operand rectangle |
| **#177** | How perf cases should be chosen at all — traced production vs the hand-picked set, with the table above. Deliberately neutral: the trace may be the flawed input |

### Open, from this work

- **Flag vs ranking — DECIDED 2026-09-04: write the ranking rule down, do not flag.** Rank the perf cases by measured ÷ their own reference, take the worst, re-rank every round — which is what run 893 did and what exists nowhere but its changelog. Flagging buys determinism and costs adaptivity, and the anchor rule that would justify it is moot for a seeded run where Phase 0 supports everything on day one. **Still to write.**
- Native's **bias cost is unexplained**. +265 µs on `(1,1,8192,7168)` for a 14 KB vector — ~60 MB of extra traffic, about a quarter of the activation. Not a vector re-read (that is ~16 µs). It tracks width linearly and not rows; the multiplier is set by rows and is independent of width. Isolating it needs per-stage instrumentation.
- Whether **translated tests gate anything**. The blind pass is "reported only, does not gate exit"; the sole gating power is the no-hang contract. If gating was intended, it is not implemented.

### F3 — closed 2026-09-04, and it was smaller than recorded

The checkpoint called this "the hardest prompt to write" — a balance between requirement 16 (preserve and extend) and requirement 15 (one-shot generation exists so the planner *can* re-plan). They do not actually conflict; they partition, and requirement 6 does the partitioning.

- **Configurations supplying no optional operand are not a design question at all.** Requirement 6 pins their program to the seed's — same buffers, same code path, same blocking — and F5 checks it mechanically by building with and without each operand and comparing. A redesign there is not a judgement call; it is a failed check.
- **Configurations supplying an operand are where the design work is.** The seed already picks its blocking from a memory budget. An operand changes what that budget must cover, so the thresholds move — measured for this op, single-read holds to ~10,048 values per row with no operands, ~6,720 with a bias, ~5,024 with both — and a shape may land in a different regime than the same shape does without the operand. The regime *machinery* is unchanged; only its input grows. Where an operand needs a scheme the seed has no equivalent of (streaming the residual rather than holding it resident is the case we found), build it as a path those configurations select, never as a replacement for the one the operand-free configurations use.

So the rule is **"design the second without disturbing the first"**, now in the prompt's `## Rules`. Requirement 11 remains the escape valve: a faster operand-free program is allowed, but only with a measurement, never with an argument.

What made it look hard was treating "re-plan the blocking" as licence to redesign the scheme. It is licence to re-run the existing solver against a budget that now knows about the operands.

## 4c. Counterpart-agnosticism — the principle, and what it cost

**The principle (ratified 2026-09-04):** an agent doing generation work must experience its task as *extending an op's capabilities*, never as reaching parity with a hand-written counterpart. Nothing an agent reads during generation may name, describe, characterise, or cite measurements taken from one. The reference implementation enters at exactly one point — the blind pass, after the work is done. Naming a counterpart turns "build the best op" into "imitate that one", independent of any cheating risk.

An audit found the pipeline did **not** uphold it: 20 blocker-level leaks from three independent sources.

### The three sources

**The successor prompt's quirks section** — fourteen entries, each shaped *"To a caller: ⟨what the counterpart claims⟩. Here: ⟨what you do⟩."* A counterpart description by construction. Two entries (X-13, X-14) stated outright that this op has no obligation from them: audit notes about another project's test suite, sitting in the planner's input.

**`feature_spec.py`** — and this was the surprise. It is a **mandated planner input**, not the perf-only file it looks like. So "re-measured from a production RMSNorm implementation", "what a DEFAULT rms_norm call achieves", "the only kind production passes" and a provenance pointer naming the op and saying it was nuked were all in front of the planner before it designed anything. Most of that text was added the same day, by this thread.

**The skill listing** — structural, and reachable by no prompt edit. The harness injects every skill's `description:` into any agent holding the `Skill` tool. The implementer holds it *legitimately* (refinements point at implementation skills), so it read, unprompted, that a hand-written op exists, that runs nuke it, and that it is recoverable from a git ref. The planner and verifier do not hold `Skill` and never saw it.

### What was done

Prompt section retitled **`## Interface compatibility constraints`**, framed as behaviours preserved for compatibility with *an op in another repository* — never named, never located. Each entry one imperative requirement. X-13/X-14 deleted; X-11 folded into layout handling; X-12 already covered by the reference-function section. X-09 **restated rather than softened** — a mixed-dtype per-channel operand must be numerically correct or refused with a message; accepting it and returning a wrong answer is the one outcome ruled out. That is stronger than the description it replaced.

Both feature_specs state targets as targets; every number, threshold and shard geometry unchanged, only attribution removed. The residual's L1 cost is now conditional ("a buffer sized to the input's and never streamed costs a second resident shard"), which leaves the design choice open rather than prescribing someone's answer. Two skill descriptions rewritten to trigger on the same intent while naming nothing. `seed.txt` dropped "the one the seed is meant to replace"; `integrity.txt` restored its own "may have been removed" hedge. `baseline_measurement.py` moved to `eval/provenance/` — scrubbing its citation while leaving a 1,188-line portrait of the counterpart inside a directory three agents are told to read would have fixed the pointer, not the exposure.

**And the recipe was fixed, not just its output.** `/golden-tests` mode C still instructed *"state what the parameter means to a caller and what the successor does with it"* — the two-part form itself. Regenerating would have written the whole thing back, which is the failure this project already recorded once. Mode C now generates the agnostic section, writes one imperative requirement per entry, carries **only entries that change what gets built**, and carries a test: a reader who has never heard of any other implementation should read the section as a plain list of this op's requirements.

Commits: `15c857a` (scrub), `b4294a3` (mode C).

### Left deliberately

- The skill **directory name** `analyze-native-gap` is injected alongside its description. Renaming it and its artifacts is the next lever if the goal is zero counterpart words in the implementer's context.
- Skill **bodies** stay saturated with counterpart language — analysis-only, so a decision rather than an oversight.
- Four other op prompts (`matmul` ×2, `flash_attention`, `rms_norm`) restate the old fp32-exclusion rule. `rms_norm`'s op is already generated; for the other two there is no evidence about whether *those* ops accept the combination, and changing their contracts on the strength of a norm-op finding would repeat the mistake being removed.

### Evidence it was framing, not integrity

`/translate-reference-tests`' description has named "reference TTNN tests… against an agentic-generated replacement" all along, and no run has ever cheated on it. That is real evidence the exposure was not exploited. It does not settle the framing half, which is why the rewrites were still worth doing.

## 4d. The blind pass changes shape for a signature-matched successor

Translation exists to bridge a signature gap. The successor closes that gap by construction, so the rewriting has nothing left to do — and the evidence gets *stronger*: an **unmodified** upstream suite passing is close to the drop-in claim itself, where a translated subset passing only proves the subset works. It also sidesteps the discovery undercount, since running files needs no case enumeration.

**"Blind" was never "hide this from the implementer"** — it is "do not task the implementer with passing these while it works; check at the end." The harness already implements exactly that with `--ignore`. Same machinery, different content.

**Feasibility, verified:** the nuke matches on *filename*, so every helper the restored files need survives (`sharded_test_utils.py`, `utility_functions.py`, `utils_for_testing.py`). `normalization/layernorm/` survives the `rmsnorm` nuke — one-way dependency, not a superstring — so every `LayerNorm*` binding the sharded suite constructs is still there. **One** public symbol needs binding; no legacy alias, no `tt_lib` path, and the lookup happens inside function bodies so a module attribute set in a conftest is seen. Filter list: ~145 distributed cases (genuinely different ops) plus **12 that upstream already skips**. About **300 cases run**. Bonus: the alias un-breaks ~50 rms cases in files that survived the nuke and are silently `AttributeError`-ing in the run tree today.

> **WRONG, corrected 2026-09-08 — and this paragraph is the root of §4f.** Every
> clause above marked "verified" was assumed. Measured: the nuke does **not** match
> on filename only — it removes whole ttnn ops, and `layer_norm` is **itself an eval
> target**, so `normalization/layernorm/` did *not* survive and neither did the four
> shared norm test files. Consequences: the `LayerNorm*` bindings the sharded suite
> constructs were gone (**97 cases**, fixed 2026-09-08 by exporting the op's own
> types); **more than one** public symbol needed binding; `utility_functions.py`
> did survive but **stale**, missing `MIX_PRECISION_TEST_IDS` that three restored
> files import; and the manifest's `origin="surviving"` for those four files was
> false, which is what crashed the pass. The one-way-dependency reasoning was
> sound; the premise it rested on — what the nuke actually deletes — was never
> checked against the tree. **TODO-9 is the direction now: port the op onto a
> declared pre-nuke tree instead of restoring into the nuked one.** Do not build
> the restore path from this section.

**The obstacle is the golden-function registry**, and its resolution is already recorded in the manifest. `get_golden_function` is the reference for essentially every restored case; the generated op is a bare function with no `.golden_function`. But **D-25 is `defer`, phase `cutover`, human-ratified** — the hook is keyed by the target's symbol, so a staging-named op cannot claim it, and the attached function is wrong about the op it describes. So registration is a cutover step, not generation work.

Which settles where the calibration lives. Our suite computes its own reference in `helpers.py` and never touches `get_golden_function`; the prompt already requires the op to export a reference **consuming every operand**. Upstream's registered golden ignores bias and residual and never forwards its epsilon — so instructing the implementer to reproduce it would contradict the prompt's own standard three sections earlier ("worse than no reference at all"). **The blind-pass layer attaches whatever calibrated golden those tests expect, for the duration of those tests.** Two things must match there and both are silent when wrong: the default epsilon, and the weight-dtype downcast (`if weight.dtype in (float16, bfloat16): input = input.to(weight.dtype)`) that models the device multiplying at the weight's precision — the tolerance budgets are calibrated against it.

**Where the plan lives:** the op's obligations are already in the prompt and need no addition; the restore/alias/filter is harness-side and agents never read it; the reasoning is here. A skill mode for "signature-matched successor runs the reference suite as-is" is worth writing *after* the mechanism has been exercised, not before.

### Built 2026-09-04 — the reference-suite layer

Opt-in per op: an op that ships a `reference_suite/` directory takes this path; every other op keeps the translated path byte-identical (verified by comparing constructed pytest argv across all 16 suites).

| file | does |
|---|---|
| `eval/reference_suite.py` | op-agnostic loader; materialises the pinned files out of git; the `has_reference_suite()` probe that makes it opt-in |
| `golden_tests/rms_norm_ttnn/reference_suite/manifest.py` | the pinned commit, the files run, the files excluded with reasons and measured counts |
| `…/filter_rules.py` | the two declarative rule groups |
| `…/golden.py` | the calibrated reference — default epsilon and the weight downcast, both documented at the definition site because both are silent when wrong |
| `…/conftest.py` | binds the symbol, attaches the golden, deselects with reasons, warns when a filter matches nothing |
| `eval/run_refinements.py` | `--ignore`s the directory in every golden-gated phase, materialises before the blind pass, hang-nodeid regexes extended |

**416 unmodified upstream cases run** — three restored files plus the rms halves of four that survived the nuke. Deselected: 12 program-factory cases (already skipped upstream), 8 whole files as distributed variants, and 744 as *not this op*.

**Both earlier estimates were low.** Surviving files carry **191** rms cases, not ~50 (`test_layernorm_sharded.py` alone is 144); the distributed group is **546** rms-flavoured cases, not ~145. And `ccl/fusion_subtests/rms_test.py` collects **zero** — its functions are not `test_*`, it is imported by a driver.

**Three decisions worth remembering:**

- **The 744 "not this op" cases are the consequential filter.** Four files carry both norms. Running them whole would add ~756 *layer_norm* cases passing against the untouched native layer_norm — inflating the pass rate with results that say nothing about this op. Scoped out as a manifest-level claim, reported as its own labelled category rather than folded into the two rule groups.
- **Only the nuked files are pinned; the surviving ones come from the run tree.** Pinning all seven produced a real ImportError — the pinned `test_layernorm.py` imports a symbol an older tree's helper does not define. A test file pinned to a different commit than the helpers it imports is version skew. This slightly weakens the "pinned = reproducible" property, deliberately.
- **A refusal is its own outcome, not a pass and not a failure.** A cell the op turns away via SUPPORTED/EXCLUSIONS reports as `REFUSED`, carrying its reason and the registry cell the runtime observer already tagged. The blind pass prints them grouped, and the line that matters names **which axis value every refused cell shares** — the question worth asking is not how many were refused but whether they cluster, and a count cannot answer that. Refusals gate nothing, move neither side of the pass rate, and are excluded from `golden_total`, so a declared refusal cannot score the op down. Cost: a new outcome travels the whole chain (conftest → results → classifier → DB → dashboard), which was most of the work.

Also fixed, because it would have silently corrupted the headline number: `db.py` classified blind rows by `test_file LIKE '%test_translated%'`, so reference-suite rows would have landed in the **golden** column. Widened to one helper shared by the SQL and the Python predicate so they cannot drift. Checked against the live database — the widened match still hits the same **203,945** existing rows and matches **zero** new ones, so nothing already recorded is reclassified.

Commit `7918c82`. Tests 494 → **518**, purely by addition.

**Known, not fixed:** the codex driver has no blind stage at all — no ignore, no hang loop — so there was nothing to mirror. The pinned sha may not be reachable in a fresh clone; the loader falls back to a filtered fetch.

## 4e. Run 998 — findings log (first seeded run, launched 2026-09-04 15:19)

Live log, appended as things surface. Clone:
`/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal`.
Config: 1 run, refinements, 3 perf rounds, JIT server `bgdepyc01:54778` x128.
Cloned at `33a6d7ea7e0` — so fixes pushed after 15:19 are NOT in this run.

**The machinery under test landed.** The seed designation rendered into the
implementer's prompt verbatim (seed path + branch), and the implementer ported
from the seed rather than generating from scratch — the thing that would have
failed silently before F1. Verified: 82–98% of the seed's lines survive, the
residual is woven into the blocking machinery (`CB_T = HAS_R ? cb_x_sum :
cb_input_tiles`) rather than bolted on, and new buffer indices sit at fresh
slots *specifically* so "the operand-free program is byte-identical to the
seed's" stays checkable. The planner also produced the first `l1_ledger.md` any
op in this project has ever produced — 236 lines, substantive.

### Findings

| # | Finding | Whose | State |
|---|---|---|---|
| 1 | `SUPPORTED` declared six operand combinations, so `weight+residual` and `bias+residual` — both legal — are refused | ours (the prompt) | **fixed, `510bc66`, next run.** The verifier did NOT catch it — predicted in advance and confirmed: its table reads `gamma_mode … same … —`, because `verify_supported` compares SUPPORTED against TARGET and the cells that ran, and TARGET also says six. A hole both sides share is invisible to a check that compares the two |
| 2 | `program_config.py:67-68` reads `bbox.end_coord`/`.start_coord`; this build exposes `.start`/`.end` | ours | open |
| 3 | `eval/metrics.py:187` calls `.max()` on a 0-element tensor — the zero-volume path correctly returns empty and the metric helper cannot measure it | ours | open |
| 4 | `op_design.md` + `l1_ledger.md` stale by one buffer | run artifact | **closed by the run itself** (`c6285a3d04`) — the implementer found the same contradiction, followed the risk row, and made the ledger match the code page-for-page |
| 5 | Some of the seed's measured-evidence comment blocks were compressed away in the port; the code they justified survives | run artifact | cosmetic |

**2 and 3 caused all 10 golden-loose failures** (433 pass / 443). Neither is an
op failure — on the zero-volume case the op passed shape, dtype and layout
checks and only the metric blew up.

### What the implementer found on its own

Four op-side fixes from running the golden suite, one substantial: the BAND
scheme gained an L1 fallback it never had — a search over activation-ring depth
then per-channel staging width, fixing a measured 62 kB CB overshoot on
`(128,8192)` fp32 ROW_MAJOR block-sharded with all three operands. Also: a
padded-last-dim rule that compared against the input's padded shape refused 48
cells at one shape alone; `inplace` returned a different Python object wrapping
the same buffer, when the contract is about identity.

### It measured seed parity itself (`c6285a3d04`)

Requirement 5 asks that nothing the seed supported regresses; requirement 6 that
an operand-free call compiles to the seed's program. The implementer measured it
without being asked, on the cells that build the same program:

```
no_gamma (32,1024) 3955->3899   (8192,1024) 84741->81493   (32,7168) 7455->7499
gamma    (32,1024) 4859->4884   (8192,1024) 89287->89544   (32,7168) 9180->9248
```

0.99x–1.04x, inside noise, everywhere. That is the drop-in claim's core
evidence, self-produced.

It also priced the operands at `(8192,1024)`: gamma 88598, +bias 1.09x,
+residual 1.42x, all three 1.59x — and reached the same conclusion we reached
independently against the native op this morning: **the residual is at the DRAM
roofline on prefill** (3 activation crossings instead of 2 = 1.50x the bytes,
1.42x the time, so no lever there) **and flat on decode**, where the combine is
latency-bound rather than byte-bound. Two independent measurements, one on the
native op and one on this one, agreeing on the mechanism.

### It built F5 itself (`b88a595d08`)

F4/F5 — the regression gate and the footprint-equivalence check — were carried
in §4 as "can be manual for the first run". The implementer wrote the second one
unprompted, and its reasoning is the same one we used: *"'the compiled program
MUST be equivalent to the seed's' is a claim a perf ratio can only ever be
evidence FOR."*

`test_program_is_structurally_the_seeds` builds BOTH descriptors from the same
tensors and compares the CB set page-for-page (`{index -> (total_size,
page_size)}` — the whole L1 footprint and the whole blocking decision made
visible) plus the compile-time args. **On the host, in 2 seconds, nothing
dispatched.** 14 geometries x {no_gamma, gamma}, one per internal scheme — row
split, interleaved width split, HEIGHT local reduce, WIDTH identity and compact,
BLOCK, the ROW_MAJOR band both ways, masked-reduce shapes, L1-tight wide ones.
All 28 identical.

The kernel comparison is asymmetric, and the asymmetry is itself the argument:
the writer must be **byte-identical** because it takes no operand and so nothing
may move; the compute kernel's seed args must be a plain **prefix**; the reader
is checked in halves because its scalars are followed by accessor blocks, so the
operands' scalars necessarily sit before them.

So F5 is no longer a manual first-run step for this op — it is a 2-second host
test in the suite. Worth promoting into the pipeline for every seeded run.

### A seeded run degenerates the refinement machinery (verifier, 2026-09-05)

The verifier's own words: SUPPORTED matches TARGET on **every** axis, so
`xfail_expected` is **0 by construction — there is no xfail bucket to be a queue
gap — and the 2:1 generality/perf cadence degenerates.**

That is structural, not a defect in this run. The refinement queue is built from
`TARGET − SUPPORTED`, and a seeded run whose prompt says "Phase 0 is not a
reduced corner" starts with that difference empty. So every refinement is a perf
refinement, and the verifier said so and moved on.

Two consequences worth carrying: the generality half of the pipeline has nothing
to do on a seeded run, and the honesty check has no gap to find — which is
exactly why finding 1 survived it. Both argue that a seeded run needs a
different verifier contract from a from-scratch one.

**Phase 0 golden: 23,348 passed / 19 failed / 98,071 skipped (INVALID cells) of
121,438 collected.**

All four refinements landed by 2026-09-06, and every one was a PERF refinement —
the degeneration above, in practice. The golden count is **flat at 23,348/19
across phase 0 and all four**, which is the correct signal: perf refinements move
device nanoseconds, not pass counts, and `refinement_type='perf'` exists so a
flat count is not misread as a stall. Refinement 4 (the prime-width granularity
cliff) reports **1.34x-10.46x** with the compute kernel untouched.

Cost through four refinements: **$199.90**, 978 turns. Wall-clock is dominated
by device testing, not agents — a single golden pass over this suite takes ~4
hours, and there is one after every refinement and every perf round.

### Where the generated op is, and what came out (as of 2026-09-07 13:36)

**The op:** `ttnn/ttnn/operations/rms_norm_ttnn/` inside the run clone —
`/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal`.
Branch `2026_09_04_1519_run1_rms_norm_ttnn`. DB run **998**. Files: the op file,
program descriptor, three kernels, `op_design.md`, `l1_ledger.md`, `changelog.md`,
`op_requirements.md`, plus the tests it wrote itself.

**Run state:** all agent phases DONE — planner, implementer, verifier, four
refinements (plus a `Refinement 4b (debug)`), and Perf 1/2/3. **$440.77, 1609
turns**, ~70 h wall-clock.

*Corrected 2026-09-08.* The blind pass was **not** running. It had already died
at 12:46, before this paragraph was written; the long-running pytest was the
graded golden run that follows it. `reference_suite/upstream/` holds three of
the seven files because the materializer crashed part-way, not because three is
all it restores. What the restore + alias + filter mechanism does and does not
do is §4f.

**Correctness:** golden **23,348 passed / 19 failed**, flat across phase 0, all
four refinements and all three perf rounds. Of the 19: 13 are OUR harness (10
`CoreRange.end_coord` in the program-config stand-in, 3 `.max()` on a
zero-element tensor), 6 are a tolerance judgement on uniform-sign inputs
(pcc 0.99990 against a 0.99995 gate). **No failing case is an op defect.**

**Speed.** Three different answers, and they must not be conflated:

| measured against | result |
|---|---|
| the previous phase (what each refinement/round reports) | refinements **1.006x** cumulative; tournaments **1.073x** overall, **1.219x** on the 19 reference shapes |
| the seed, operand-free | **0.99–1.04x** — the point is that it is identical there |
| native, on the reference shapes | **beats it on 17 of 19**; 0.07x–0.99x, the two exceptions 1.02x/1.04x (noise) |

The large margins vs native are **inherited from the seed's architecture**, not
earned this week — chiefly that native runs the few-rows cases on ONE core while
this op parallelizes them. The refinements added ~0.6% on top; the tournaments
~7%. Note the effort split is inverted against that: refinements ~$75,
tournaments ~$154.

**With optional operands present, it beats native on 4 of 5 reference cases**
(0.07x, 0.52x, 0.88x, 0.97x) and ties the fifth. Thin coverage though — only 5
of 19 reference shapes carry operands, and all five carry all three.

**Unexamined regressions, twice.** Diffing per-phase timings (nothing does this
today — #179): **214 shapes** 5–21% slower after the refinements, clustering on
BLOCK_SHARDED across every operand mode; **107 shapes** >13% slower after the
tournaments, clustering on INTERLEAVED + ROW_MAJOR. Both clusters name a
discriminator. Neither was reported or judged.

**Perf 1 -> Perf 2 was 0.998x** — the second round earned nothing overall (best
2.74x, worst 0.74x: churn). Evidence for TODO-3.

### TODO-LIST — prompt / agent fixes this run exposed (ranked, 2026-09-07)

*Items are numbered TODO-1..TODO-13. Do not confuse with §1's lettered requirement
groups. TODO-1..5 are prompt/agent fixes this run exposed; TODO-6..13 are eval-infra
gaps the reference suite exposed (§4f), and they are the ones that decide whether this
class of defect is caught automatically next time.*

**TODO-1. ~~The verifier's contract~~ — DONE, submodule `812c68f`.** Landed: the thin-or-empty gap rule (generic), the seed-as-baseline framing and the whole-golden-set no-regression rule (seeded-only), the honesty-check caveat redirected to a spec finding, and the 2\*\*N presence rule in `/golden-tests`. A refusal probe was built and then withdrawn — the fix is that the SPEC lists every legal value, not that the verifier hunts for omissions; the probe's cost was not worth a detector for a bug that should not exist. Original statement follows.

**TODO-1 (as specified).** Two triggers, and they are ORTHOGONAL:
whether a seed is designated, and whether `TARGET − SUPPORTED` is empty. "Phase
0 is not a reduced corner" is a line in *this op's prompt*, not a property of
seeding — a seeded run can aim at a wider TARGET than its seed covers and leave
a real gap, and a from-scratch run can land Phase 0 fully supported. Key on each
condition independently.

*Generic — keyed on the condition, not the run type:*

1. **Probe what SUPPORTED refuses.** Call values the declaration rejects. Free —
   a refusal happens before any device work. Two outcomes, two actions:
   op accepts + SUPPORTED refuses → file a **correction** (one line in the op
   file; acceptance is "declared and the cell passes", not "feature built");
   SUPPORTED claims + op fails → the existing `supported_fail` over-claim.
   This is the check that would have caught the six-vs-eight hole.
2. **Handle an empty `TARGET − SUPPORTED`** — keyed on the emptiness itself.
   Perf-first queue, reported as an assertion satisfied rather than a cadence
   that "degenerates".

*Seeded-only:*

3. **Structural equivalence as a FAST PATH, not the requirement.** The
   requirement is no functional and no perf regression; identical buffers + CT
   args is a cheap *sufficient condition* for both, so where it holds the cell is
   settled with no device time (28/28 cells, 2 seconds, host-only on this run).
   Where it differs, that is not a failure but a **bill**: that cell owes the
   seed's tests passing and its time within noise. The implementer built exactly
   this unprompted — one commit is *"seed-parity test carries the derived-arity
   divergence, with its measurement"*. The verifier must REQUIRE it, not hope.
4. **No-regression against the seed** (req 5) — correctness, plus timing where
   structure diverged. **One-sided: faster passes silently.** Nobody ran this.
5. **The baseline is the seed, not the target** — stated in req 5, absent from
   the verifier's contract. A framing statement rather than a check.

*Not the verifier's job* — "the op accepts a value the suite never generates".
Finding that means enumerating the cross-product against what the suite emits;
too expensive mid-run, and it is a **spec-authoring bug** anyway. It belongs in
`/golden-tests` mode C, which already holds both TARGET and the presence map at
authoring time and can force either an axis value or a loose case for free.
Note this was the real shape of our own bug: fixing SUPPORTED alone would have
left `weight+residual` legal and never once exercised — the prompt fix AND the
ten loose cases were both needed.

*Req 12 ("preserved advantages") is NOT a separate check* — it is the one-sided
half of no-regression, and it is a rule for **every stage**, not just the
verifier. A perf round that speeds up the operand-free path is a win; req 11
already licenses it ("justified by measurement"). **Flagged to revisit.**

**TODO-2. The pipeline's effort split is inverted for a seeded run.** Four
refinements cost ~$75 and delivered **1.006x**. Two tournaments cost ~$154 and
delivered **1.073x** overall, 1.219x on the reference shapes. If a seeded run has
no generality gap, the generality queue is close to dead weight — consider going
straight to tournaments.

**TODO-3. Fixed round counts waste money.** Perf 1 -> Perf 2 measured **0.998x** —
the second round earned nothing overall (best 2.74x, worst 0.74x: churn). Stop
when a round comes out flat rather than running a fixed three.

**TODO-4. A static-analyzer check for guard-tests-wrong-flag.** Third sighting in
this project: `static_assert(IS_TILE != 0, "TILE per-channel operands only")`
read the ACTIVATION's layout flag, not the operand's, and broke 48 cells at
build time. The message named the right rule; the condition read the wrong
variable two arg slots away. Previously seen as a misspelled `#define` making a
condition permanently true. A guard whose text and test disagree is invisible to
review — but a checker can compare the asserted symbol against the message.

**TODO-5. The last refinement's timings never got ingested.** `Refinement 4` has no
`device_kernel_ns` rows, so its claimed 1.34x-10.46x cannot be verified from the
DB and the cumulative measurement has a hole in it.

**TODO-6. The delta manifest is a specification with no acceptance test.**
The biggest gap this run exposed. `D-06` was ratified **adopt** with 60 usage
sites; the op adopted the *reading* of two config fields off a duck-typed object
and never exported the types, so no caller could construct the argument. Nothing
anywhere compares the ratified manifest against the shipped op — the verifier
builds its queue from `TARGET − SUPPORTED` and never opens `native_gap.json`.
Every ratified delta is therefore advisory. Wanted: a mechanical check that each
`adopt` item is present in the successor's declared surface. Signature and type
deltas are fully checkable; semantic ones need a named test.

**TODO-7. The golden suite may fabricate what the op lacks.** Two independent
instances in this run. `program_config.py` defined the two config types as
"stand-in" dataclasses; `helpers.py` built the row-major weight in the one
spelling the op happened to accept. Both times the suite validated the op
against itself and went green — 32 dedicated weight cases, all passing, all
wrong. Wanted: if the suite must invent a type or a spelling to place a call,
that is a missing op surface, not a fixture. `/golden-tests` should say so, and
a stand-in should be a gate failure, not a docstring apology.

**TODO-8. The only outside witness runs last.** The reference suite sits after
every refinement and every perf round. Both defects above were visible in its
FIRST file. Three days of refinement and two perf tournaments were built on top
of an op that no real caller could call. Wanted: a slice of the reference suite
at phase 1, gating. Cheap — one file, ~90 cases, 30 s on device.

**TODO-9. Run parity on a declared pre-nuke tree, not by restoring into the nuked
one.** *Rewritten 2026-09-08 — this is the direction to pursue; the original
statement is at the end of the item.*

The restore direction is upside down. Porting main's tests INTO a nuked tree means
every claim about what the nuke left behind can go stale (it did — §4f), a
restored test drags its helpers with it, and test and helper can come from
different commits (they did). Invert it: **port the generated op ONTO a pre-nuke
tree and run parity there.** Nothing is restored, so nothing can be stale.

What that tree gives, measured on `1fa25618dfe^` (this branch's pre-nuke state):

* all seven reference files and every helper present, at **mutually consistent
  versions** — the skew that broke us cannot exist, because there is only one
  version of each;
* **native `rms_norm` present**, so the alias becomes a substitution rather than a
  creation, and a same-tree A/B against native comes free;
* **`ttnn.LayerNormShardedMultiCoreProgramConfig` present**, so `ALIASED_TYPES`
  becomes unnecessary and the op is exercised against the REAL config object
  instead of a Python stand-in — strictly stronger evidence than the 2026-09-08
  fix;
* the manifest collapses to the alias plus the case filter: no `origin`, no
  restore step, no partial-write hazard.

**The graft is cheap.** The op is pure Python host-side (op file, program
descriptor, config types) plus device kernels that JIT-compile at test time —
**no host C++, no CMake**. So: copy the op dir, register the package, symlink
`eval/`. One build per counterpart tree, cached across every run and every op.
The kernels then compile against the COMPARISON tree's helpers, which is why the
next paragraph is the load-bearing one.

**The counterpart SHA is declared, never derived.** There is no single "nuke
commit" to take the parent of: a nuked branch may be *rebased* onto
`llk_helper_library` repeatedly (many nuke commits, no meaningful parent) or
*forked* from it and nuked once. Picking the nuke out of a log is exactly the
inference req 14 forbids. So seeded mode **requires** the counterpart SHA — no
default, no heuristic. Best home for it is a provenance file on the nuked branch
itself, updated by the rebase workflow, with a CLI flag overriding; that keeps the
fact next to the thing it describes and keeps a stored run auditable.

**A declaration is not equivalence — check it, and fail hard.** A declared SHA only
makes the assumption visible. Verify the counterpart is helper-equivalent to the
run tree, and **derive the surface to compare from the op's own kernel includes**
(`ttnn/cpp/ttnn/kernel_lib/...`, the `api/compute` + `api/dataflow` roots, the
`ckernel_sfpu_*` headers) rather than a maintained list of paths — that
generalizes to any op. On mismatch, **fail** in seeded mode: a no-regression claim
measured against different helper behaviour is a different claim, not a weaker
one. Record the SHA and the per-path result, so parity numbers stay attributable
to a tree — run 998's are not.

Measured today, this check has an immediate positive: `llk_helper_library` has
diverged from `dnijemcevic/agent_eval_new` in five `kernel_lib` files, and the
EVAL branch is the one ahead (a Quasar guard, a broadened fp32-`Accurate`
condition). Signatures are identical — the `.hpp` diff is comment-only — so it
compiles and behaves differently. Under the rebase construction this check will
fail routinely, and that is information: the branch needs a rebase before its
parity claims mean anything, which is what `/llk-helper-rebase` is for.

**What it gives up.** The reference tests are the counterpart's, not main's tip, so
"parity with main" becomes "parity with the lineage we forked". Honest rather than
weaker: the op cannot run on main at all today, since its helpers are not there —
parity with main's tests was always aspirational. Drift is small in any case: on
this branch `test_rms_norm.py` and `test_layer_norm.py` are IDENTICAL to the
pinned ref, the other rms files differ by 3-6 lines, and only the nightly layernorm
trio has moved. A pinned-ref restore can stay as an OPTIONAL top-up for fresher
cases, rather than the mechanism the whole suite rests on.

**Also settled by this:** the open helper-restore decision at the end of §4f is
retired — nothing gets restored. And the comparison tree should run the op's own
golden suite before the reference suite: if the op does not pass its own suite
there, its parity numbers mean nothing.

*Original statement (still true of the restore path, if it is ever kept):*
`origin="surviving"` was wrong for four files and nothing checked until the write
crashed half-way; the partial write then silently contaminated the graded run
(TODO-11). Wanted: verify every declared path before writing anything, and write
all-or-nothing.

**TODO-10. The blind pass neither gates nor recovers.** Its crash is non-fatal,
so the run reported **PASS** with zero reference evidence while 112 reference
cases were failing. There is also no standalone entry point, so it cannot be
retried without re-entering the whole refinement driver. Wanted: a hard failure,
and a `--blind-only` mode.

**TODO-11. Two suites share one number.** Nothing excludes `reference_suite/`
from the graded pass — the blind-pass code says outright that the full golden dir
runs with no `--ignore`. So a successful run folds all 416 reference cases into
the headline golden count. `23453/121663` is not one population. Wanted: separate
counts, and cleanup or exclusion of `upstream/` after the blind pass.

**TODO-12. Self-reflection is gated on the blind pass.** `run_eval.py:680`
returns early when `golden_blind_final/test_results.json` is absent, so the one
agent whose job is finding exactly the defects above never ran on the run that
most needed it. Wanted: reflect on what exists rather than requiring the blind
pass to have succeeded.

**TODO-13 (small). `run_safe_pytest.sh` ignores `PYTHON_ENV_DIR`.** It hardcodes
`./python_env`, so in a clone it silently falls back to system Python and dies on
a conftest import. Every hand-run probe against a clone hits this.

### Lesson worth keeping beyond this run

`TARGET` says what we **test**; `SUPPORTED` says what **works**. Finding 1 is
what happens when the first silently answers the second — a decision about test
budget (sweeping two more combinations costs +27,440 device tests) became a
decision about capability, and nobody made it. A distinct compiled program needs
*some* coverage, not the full cartesian: ten loose cases, not 27,440.

## 4f. The outside witness spoke, by accident (2026-09-08)

The reference-suite layer's first real execution. It did not run as designed,
and the way it failed is more instructive than a clean pass would have been.

### What actually happened

The blind pass died at 12:46 on its first statement, before any test ran:
`test_layer_norm.py` is declared `origin="surviving"` and is not in the tree.
**The nuke removes whole ttnn ops, and `layer_norm` is itself an eval target**, so
it took the entire shared norm test cluster with it. All four files the manifest
calls "surviving" are gone — 191 of the suite's 416 cases.

The manifest's *reason* for reading those files from the tree was correct and is
worth keeping: pin a test to the ref while its helper comes from the tree and you
get a version-skew import error. Measured — the tree's
`nightly/.../utility_functions.py` does not define `MIX_PRECISION_TEST_IDS`, which
three of the four files import. The **conclusion** was inverted: the fix is to
pin the test *and its helper together*, not to pin neither. The ref's helper is a
strict superset of the tree's (same sixteen wrappers, plus the mix-precision
constants), so taking both from the ref costs nothing.

Then three things compounded:

1. `materialize()` writes the `nuked` files first, so `upstream/` was left
   holding the three rms files — a **partial write**, no rollback.
2. A non-zero refinement exit is non-fatal, so the driver fell through to the
   graded golden run.
3. Nothing excludes `reference_suite/` from the graded pass, so that run swept in
   the three orphaned files, bound the alias through their conftest, and **ran
   225 upstream cases for real** — folding them into the headline
   `PASS 23453/121663`.

Self-reflection then skipped itself, being gated on blind results that did not
exist. So the run reported PASS, with two populations mixed into one number and
no self-reflection pass, while 112 reference cases were failing.

Net: we got half a blind pass by accident, and it immediately found two defects
that three days of refinement and two perf tournaments had not.

### What it found — and both were already in the manifest

**`D-10`, weight shape (15+ cases).** The manifest nailed this: kind
`domain-narrower`, direction **incompatible**, both sides quoted, and a
prediction of the exact failure — *"a target-shaped (Wt, 32) row-major weight has
last dim 32 and is therefore refused whenever W != 32."* Ratified **adopt** on
2026-09-02. The op refused it anyway.

Where it was lost is one dimension. The target's rule, measured, is
rank-agnostic: `padded last dim == tile width` and
`physical_volume / tile_width == input padded width / tile_width`. The manifest's
**evidence quote is that rule, verbatim and correct.** Its **prose summary**
compressed it to "(Wt, 32)", which reads as rank 2 — and the prose is what
travelled: into the prompt ("shape (Wt, 32)"), into the harness
(`padded.reshape(Wt, 32)`), into the op (`len(shape) == 2` as a hard conjunct).
Every stage agreed with every other, so 32 dedicated golden cases passed. Only
upstream spells it the way real callers do: `reshape(1, 1, w // 32, 32)`, rank 4.

**Why the summary sentence, and not the quote, is what matters.** G14 deliberately
strips the target's source before the delta travels — that is what keeps the target's
code out of the prompt. So the summary sentence is the ONLY thing anything downstream
ever reads, and it was the one field nothing checked.

**`D-06`, the program-config type (97 cases).** Enumerated too, with both type
names and 60 usage sites, ratified **adopt**. The op adopted reading
`subblock_w` and `inplace` off a duck-typed object and never exported the types.
The golden suite papered over it with "stand-in" dataclasses — field-for-field
identical to native, docstring included the tell: *"the same cases run unchanged
once that object exists."* It never existed.

### The pattern behind all of it

Both defects have the same shape, and so does the operand-combination finding above:
**the specification described less than the target allows, and everything built from
it inherited that.**

The prompt, the tests and the op all come from the same document. So they agree with
each other by construction — they cannot catch this, however carefully anyone reads
them. Only tests written by people who never saw our specification can, which is what
upstream's own tests are.

Two things follow. A passing golden suite proves nothing when the suite was written
from the same sentence as the op (TODO-7). And upstream's tests have to run early, not
last (TODO-8).

### Ledger — distance to a green reference suite

416 cases, all accounted for:

| block | cases | closes how |
|---|---|---|
| passing | 193 | — |
| failures | 16 | op work: `uneven_multicore_logical_width` (12), `width_non_rectangular_grid` col-major (4) — both cross-core width distribution |
| refusals | 16 | **already fixed** in submodule `510bc66` (six of eight operand combinations); this run predates it, closes on regen |
| never executed | 191 | the four shared files — **not** by restoring them: they come free on a pre-nuke comparison tree (TODO-9) |

**We can currently see 225 of 416.** The remaining distance is unknown and may
exceed everything found so far.

### Fixed 2026-09-08

In the clone unless noted. All measured, none asserted.

| fix | where | evidence |
|---|---|---|
| `per_channel_form` keys on element count, not rank | op descriptor | upstream `test_rmsnorm.py` **29/29** (was 14/29); llama 15/15; the 38 rank-2 loose cases unregressed |
| the two program-config types moved onto the op's public surface and exported; golden suite imports them; `ALIASED_TYPES` added to the manifest and bound in the reference conftest | op + suite + manifest | sharded reference file **82 pass** (was 9) |
| `CoreRange.start`/`.end` (Python names; the C++ members are `*_coord`) | suite `program_config.py` | 7 of the 10 phantom failures |
| zero-element guard in `compute_metrics_torch` | `eval/metrics.py` — **clone AND submodule** | the last 3; 518 eval unit tests green |
| the skill must state a rule, not one example of it | `/analyze-native-gap` SKILL.md, 5 edits | see below |

**The 10 "regressions" the verifier reported for three consecutive phases were
all harness defects.** Zero were op bugs. `supported_fail=10` stood unexamined
through Perf 1, 2 and 3.

**Skill fix (`/analyze-native-gap`, 5 edits, ~8 lines):** write a constraint as the
rule the target's code actually tests, never as one shape that happens to satisfy it
— in Step 4 where items are built, in the schema comment, and modelled in the
aspect example. Plus the audit half: constraint items are now in audit scope, and
the `wrong` verdict explicitly includes **true but narrower** ("name one input the
source accepts and the prose excludes"). G18 already requires every `wrong`
resolved, so no new gate was needed. Deliberately not built: a `rule`/`free`
schema block with witnesses — witnesses are over-fittable (two examples read as
"rank <= 3") and the predicate is the specification, so examples belong on the
analysis side of the G14 strip line, if anywhere.

### Retired — restoring a nuked test *helper*

Left open for one day and then dissolved. The question was whether the
materializer may write outside its own directory to restore
`nightly/.../utility_functions.py`, since `test_layernorm.py` imports it by full
module path and the tree's copy lacks `MIX_PRECISION_TEST_IDS`.

It does not arise on a pre-nuke comparison tree, where test and helper are the
same tree's own matched pair. See TODO-9 — that is the direction, and nothing is
restored. Kept here only so the reasoning is not re-derived: the version skew was
real and measured, but it was a symptom of mixing two commits, not a thing to
engineer around.

## 4g. Session inventory and the three buckets (2026-09-08)

### What is an instrument and what is output

A change inside a run clone is a **diagnostic instrument, not a fix.** The clone's op
is disposable — a regen replaces it — so a clone edit's only value is what it reveals.
Carrying an op edit back would hand-patch the very thing whose automatic generation is
the point, and would destroy the test of whether the prompt/skill fixes took.

| location | files | status |
|---|---|---|
| clone, op (`ttnn/.../rms_norm_ttnn/`) | 3 modified + `program_config.py` new; +38/-23 | **instrument — do not carry back** |
| clone, eval submodule | 5 files, +104/-57 | 2 worth promoting (below) |
| main tree, eval submodule | `eval/metrics.py`, `skills/analyze-native-gap/SKILL.md`; +23/-6 | **the actual output** |
| main tree | this journal | output |

### Every clone edit, and what it revealed

Listed because the earlier ones were made without being enumerated at the time.

| # | edit | revealed | infra outcome |
|---|---|---|---|
| 1 | `per_channel_form` keys on element count, not rank (+ docstring) | manifest prose compresses a rule into an example | skill fixed |
| 2 | the two program-config dataclasses moved onto the op's public surface, exported from `__init__` | a ratified `adopt` delta (D-06) shipped half-implemented | TODO-6, open |
| 3 | golden suite imports those types instead of defining stand-ins | the harness can fabricate what the op lacks | TODO-7, open |
| 4 | `CoreRange.start`/`.end` in the suite's `program_config.py` | 7 of the 10 phantom "regressions" | none — suite bug |
| 5 | `ALIASED_TYPES` in the manifest + bound in the reference conftest | the alias list was one symbol where callers need several | subsumed by TODO-9 |
| 6 | zero-element guard in `compute_metrics_torch` | shared metrics crashed on a legal shape | **landed in submodule** |
| 7 | four `origin` flips + `RESTORED_HELPERS` | the nuke's footprint was assumed, never measured | §4d corrected; TODO-9 |
| 8 | `materialize()` verify-all-then-write + helper restore | a partial write silently contaminated the graded run | **promote** |
| 9 | grid coercion in the config shim (`(x,y)` -> `CoreCoord`) | my own stand-in diverged from the real bound type | TODO-7 again, self-inflicted |
| 10 | `gamma_residual` + `bias_residual` in `SUPPORTED` | spec named six of eight legal combinations | fixed pre-session (`510bc66`) |
| 11 | `inplace` refusal deleted | quirk verdicts state prohibitions, not behaviour | skill fixed (`call_outcome`) |
| 12 | `max(0, min(shard_w_t, Wt - w_start))` at 2 sites | op invents its own quirk of the class the register catalogues; and the band path already guarded the same edge | open, no gate looks for it |

Edits 6 and 8 are op-agnostic and are the two worth promoting. Everything else stays.

### Fixed — in the eval system, so future runs get it

**1. The gap analysis must write down the RULE, not an example of it.** It had said the
weight arrives as shape `(Wt, 32)`. The rule in the target's code is "last dimension is
32, and the total element count matches the width", which allows many shapes. Writing one
example instead of the rule silently added a requirement that was never there, and the
generated op then rejected the shape real callers use. The skill now says to write the
rule, in the three places that matter: where items are built, the schema comment, and the
worked example.

**2. The reviewer may now fail a description for being too narrow.** Before, it asked only
"is this description true?" — and `(Wt, 32)` *was* true, just narrower than the code it
quoted. It now has to ask "does this description allow everything the code allows?" and
name one input the target accepts that the description excludes.

**3. Notes about the target's quirks must say what a CALL DOES, not what to avoid.** The
note about `inplace` said "don't silently accept a value you throw away". The op read that
as "reject the call". What it meant was "accept it, ignore it, and document that". There is
now a required field saying plainly whether such a call **works** or **errors** — a field,
not a sentence, so there is nothing left to interpret. It must also agree with the
disposition of the item it feeds.

**4. Shared test code no longer crashes when an op returns an empty result.**
(`eval/metrics.py`.)

**5. §4d's paragraph claiming we had verified what the nuke deletes is corrected.** We had
not. That one unchecked assumption caused everything in §4f. It now carries a warning not
to build the restore path from it.

### Remaining — we know what to do

**6. Nobody checks that the op did what the gap analysis told it to do.** It disobeyed four
times and nothing noticed (the weight shape, the config types, `inplace`, and the operand
combinations).

**7. The review step asks the wrong question.** It asks "is the op honest about what it
claims to support?" It never asks "does the op support everything it was told to?" So an op
that refuses perfectly valid calls passes review — which is exactly what happened to 120
cases. Same missing check as 6, from the other side.

**8. The test suite is allowed to invent things the op is missing.** It faked the config
objects the op should have provided, and it built the weight in the one shape the op
happened to accept. So the tests agreed with the op and everything looked green while being
wrong. A stand-in means a missing piece of the op, not a fixture.

**9. Upstream's own tests run last, after everything else.** They found every problem above
in their first file. Run a slice of them early instead — one file, ~90 cases, 30 s.

**10. Better way to run the parity tests:** put the generated op into a branch that still
has all the tests, rather than copying tests into the stripped-down branch. Direction
settled (TODO-9), not built. The counterpart commit is declared, never guessed, and checked
for matching helpers before any number is trusted.

**11. When the reference-test step crashes, the run still reports PASS.** It should fail,
and it should be re-runnable on its own.

**12. Two different test suites get added into one score.** Report them separately.

**13. The final review agent is skipped whenever the reference step didn't run** — exactly
when it is most needed.

**14. One op bug we understand:** the op rejects the caller's block size by comparing it
against an internal number it worked out itself, *after* having already approved that block
size. The right fix — let the internal number accommodate the caller, per A5's "honoured,
never clamped" — touches the L1 budget logic.

### Remaining — we do not know

**15. Why 87 tests still give wrong answers.** Most sit in two sharded mixed-precision
groups, so probably one or two causes, but nobody has looked. They were invisible until the
`inplace` refusal came off, which is this session's other lesson: **a refusal hides
untested code.** Refusing valid calls costs the refused cases *and* every scrap of evidence
about the code behind the refusal.

**16. Whether the fixes above actually work.** Only a fresh run answers it. Every clone edit
listed above is a bet that the next run will not need it. If it does, that fix failed, and
that is the specific thing to examine.

**17. Whether the tuning rounds were wasted.** Four correctness rounds and three performance
rounds ran against an op real callers could not call. Nobody has asked what they would have
chosen otherwise.

**18. Whether the plan in 10 works in practice.** It looks cheap on paper — the op is pure
Python on the host with kernels compiled at test time, so no rebuild. Untested.

**19. How many more "we told the op, the op did something else" cases exist.** Four were
found by accident, from a single upstream test file. No systematic sweep has been run.

**20. Whether this is this one op or the method.** We have done one op.

## 4h. Interface quirks a drop-in must accept — and what should change instead

Recorded 2026-09-09. These are properties of the TARGET's interface that a drop-in replacement
has to accept because callers already pass them. Each is, on the evidence, worth fixing in the
target rather than propagating. The manifest catalogues them as `inconsistencies` (X-nn); this
section groups them by **what the remedy is**, which the manifest does not.

### Accepted and never read — delete the field

* `compute_kernel_config.packer_l1_acc` — destructured by every factory, referenced by none.

### Validated, then a different source is used — delete the check or honour the value

* `compute_with_storage_grid_size` — range-checked on entry, then placement comes from the
  shard. Reads as a placement contract; is not one.
* `memory_config` under `program_config.inplace` — the output IS the input, so a requested
  placement describes an allocation that never happens. Two independent arguments where one
  silently overrides the other. Either refuse the pair or state the aliasing.

### Exposed but pinned — the parameter can express nothing

* `program_config.block_h` / `block_w` — presented as free, but equality assertions force them
  to restate the shard's own geometry.
* `program_config.use_welford` — a field on both config variants, asserted OFF for this op.

### Exposed but REDUNDANT — a live knob pointed the wrong way

* `program_config.subblock_w`. **Measured 2026-09-09, all three trees:** it sizes **zero** L1
  buffers. Native's every buffer comes from `DFBSizeParams::compute()`, which has no such
  field; the successor's 40 CB allocations and its `_cb_block_bytes()` never mention it. On a
  real pinned case (`(1,1,64,1536)`, width-sharded, 6 cores, fp32 acc) the per-core arena is
  141,312 B and sweeping the knob 1 → 4 changes **0 of them** — 0.000% of the arena and of the
  1.5 MB budget. What it really trades is DST register occupancy against per-block fixed
  overhead (one init, one format reconfig, one CB reserve/push amortized over the block).

  So the degree of freedom is real — 1.59x on the seed's pass-B stage — but the **knob** is
  redundant: the op derives the largest divisor of the resolved width chunk under the DEST
  limit, and that default MEASURED 24,497 ns against 25,513 for the hand-tuned value the spec
  held up as the prize. The caller is strictly worse informed: the ceiling is a register-file
  limit they cannot see, and the divisibility constraint comes from a chunk width derived from
  an L1 budget they also cannot see.

  Distinct remedy, which is why it gets its own heading: the inert fields above can simply be
  deleted. This one is consumed and effective, so the fix is a default-is-better recommendation
  — or a way to ask the op what it would have chosen — not removal.

  **Provenance:** absent from the seed entirely (`program_config` was "reserved; ignored when
  None", taken and never read). The seed's internal equivalent was derived IN THE KERNEL, never
  chosen. Parity asked only that the field stop being swallowed (D-06 `adopt`) and that zero be
  refused (X-06); **nothing asked for a new tuning surface.** The prompt's "HONOURED, never
  clamped" is not achievable on plans that re-chunk the width, and the op silently takes the
  nearest legal value there — correct behaviour, false claim. Fix the sentence, not the clamp.

### Documentation that disagrees with the code — the dangerous group

* The documented weight/bias dtype table is enforced on the row-major branch and unchecked on
  the tiled one.
* A float32 weight against a bfloat16 input is accepted and produces wrong numbers rather than
  being refused.
* The entry point refuses row-major input while the shared implementation underneath permits it.
* The registered reference implementation ignores bias and residual and never forwards epsilon,
  so anyone comparing against it compares against the wrong formula.

The first three groups are dead weight a future caller trips over. The last group is where the
documentation and the code disagree and the code wins silently.

## 4i. The gap between "passes its own tests" and "passes the target's" (2026-09-09)

Written for someone who was not here. The op passed ~23,300 of its own generated cases and had
never been run against the target op's own test files. When those finally ran, 93 of 416 failed.
Ten distinct defects, in the order they were found. Each entry says what the thing IS, what was
wrong, and **why 23,300 passing cases could not see it** — that last part is the reusable half.

### 1. The weight had to be a flat two-dimensional tensor

*What it is.* The weight is a per-channel scale: one number per column of the input, so 2048
numbers for a 2048-wide input. The hardware reads in 32-wide chunks, so the target lets a caller
hand those 2048 numbers over **folded** into 64 rows of 32 rather than one row of 2048. Same
numbers, same order, different physical arrangement.

*Wrong.* The op accepted the folded form only if the tensor was literally 2-dimensional,
`(64, 32)`. Every real caller passes `[1, 1, 64, 32]` — the same thing with two size-1 dimensions
in front. Refused.

*Why the tests missed it.* The spec described the folded form by **writing down one example
shape**, `(Wt, 32)`, instead of the rule ("last dimension 32, total count matches the width").
A shape written with two numbers implies two dimensions whether the author meant it or not. The
tests were written from the same sentence, built the 2-D form, and passed. 32 dedicated cases,
all green, all wrong.

### 2. The op accepted a tuning object it gave callers no way to build

*What it is.* The target takes an optional tuning object where the caller says how to split the
work: core grid, block sizes, and whether to write the answer back into the input. You have to
construct it — `ttnn.LayerNormShardedMultiCoreProgramConfig(...)`.

*Wrong.* Our op reached for two fields on whatever object it was handed (`.subblock_w`,
`.inplace`) without ever providing a type. That works if someone hands you one; it is useless to
someone who has to create one. So the argument was unreachable in practice.

*Why the tests missed it.* **The test suite invented its own look-alike Python class.** So the
tests could place the call, and no real caller could. A missing piece of the op's surface was
supplied by the harness.

### 3. The type I added then rejected `(8, 8)`

*Wrong.* When I supplied the missing type I wrote a plain Python class with the right fields.
The target's version accepts its core-grid argument either as a proper `CoreCoord` **or** as a
plain pair like `(8, 8)`, because its C++ binding converts automatically. Mine did not, and 120
cases died on it until I added the conversion.

*Why the tests missed it.* They didn't — this was mine, made the same afternoon. It is the
sharpest available argument against a hand-written stand-in for a bound type: I copied the
fields and missed the behaviour.

### 4. Two of eight operand combinations were refused

*What it is.* Three optional inputs (weight, bias, residual) make eight on/off combinations.

*Wrong.* The spec listed six. So the op declared six, and two perfectly ordinary calls —
weight+residual, and bias+residual — were refused. Nobody decided that; it fell out of a
decision about how many test cases to run.

*Why the tests missed it.* The combinations were **typed out by hand** rather than stated as
"all 2^N are legal". The tests enumerate the same six. (Fixed in the prompt, submodule
`510bc66`.)

### 5. `inplace` together with `memory_config` was refused

*What it is.* Two independent arguments. `memory_config` describes where a tensor lives — memory
kind, layout, shard geometry — and as an argument it is a request for **the output tensor's**
placement. `inplace=True` is a field on the tuning object meaning "write the answer into the
input and hand that back". Note there is no output-tensor argument anywhere in the signature; the
op always returns its own result.

*Wrong.* Under `inplace` no output is allocated — the result IS the input, whose placement was
fixed when it was created. So the request has nothing to apply to. The target accepts it and
ignores it. Our op compared the request against the input's actual placement and raised when
they differed. 120 cases.

Two errors stacked. In the failing cases the caller's config was not even substantively
different: it gave layout and buffer type but **left the shard geometry unset**, and our
comparison read "unset" as "different". The target treats unset as "use the input's".

*Why the tests missed it.* The note describing this quirk said what **not** to do ("silently
accepting a value it discards is the part not to copy") rather than what a call should do.
Refusing the call also stops the silent acceptance. The op chose refusing, wrote its own test
asserting the refusal was correct, and the review step read the same sentence and passed it.
(Fixed in the skill: verdicts now carry a machine-readable `call_outcome`.)

### 6. Spare cores were given a negative amount of work

*What it is.* Work is split across a grid of cores, each taking a slice of the tensor's width.
In the code: `Wt` is the whole tensor's width counted in 32-wide tiles, `shard_w_t` is how many
of those tiles one core's slice holds, and `w_start` is where this core's slice begins.

*Wrong.* Each core's extent was `min(shard_w_t, Wt - w_start)` — "my share, or whatever is left,
whichever is smaller". When the grid had more cores than there was width to go round,
`Wt - w_start` went **negative** for the leftover cores, and `min` picked it. That number is
handed to the hardware as an unsigned value, so negative became enormous and the call crashed.
The fix is `max(0, ...)` — a spare core owns zero tiles, never a negative count. The band-based
path one screen above already had that guard; the tile-based path did not.

*Why the tests missed it.* No case ever used a core grid larger than the work needed. The suite
always sized the grid to fit.

### 7. Column-major shards read the wrong data — 73 cases, the biggest single fix

*What it is.* When a tensor is spread across cores, its shard spec includes an **orientation**:
walk the cores across rows, or down columns.

*Wrong.* The op read that field in two places, both of which only compared two shard specs for
equality. It never used it to decide which core gets which piece of the tensor. So on any
column-major shard, every core normalized a slice belonging to someone else, and the statistic
multicast went to the wrong family of cores.

*Why the tests missed it.* **Orientation is not an axis in the spec at all.** It is reachable
only through a per-case override that nothing sets, and the shared shard-building helper defaults
to row-major. Zero of ~23,300 cases could have hit it.

### 8. The op's own intermediate values were kept in the compressed input format

*What it is.* `bfloat8_b` is a block-compressed format: sixteen numbers share one 8-bit exponent.
It is a reasonable **storage** format for a tensor a caller chose it for.

*Wrong.* The op held its own intermediates — the running sum, the squares, the normalized block —
in that same compressed format, so each of three stages paid a fresh block-float rounding on
values the kernel had just produced. The target holds intermediates at full width regardless of
the input's format.

The fix promotes them to **bfloat16, not float32** — deliberately: bfloat16 is already at the
accuracy floor here (1.013e-2 against float32's 9.98e-3, i.e. the remaining error is the
input/output quantization the caller asked for) and costs 2048 rather than 4096 bytes a tile.
Non-block-float builds are byte-identical to before.

*Why the tests missed it.* Not coverage — the case **ran**. Our accuracy gate for `bfloat8_b`
(PCC 0.99, relative RMS 0.10) was about **6x looser** than the target's own gate for the same
thing (relative Frobenius 1.6e-2). The real error, 1.708e-2, was outside the target's gate and
comfortably inside ours.

*Cost.* The only fix that touched a hot path, and the only one measured: net 0.996x over ten
`bfloat8_b` cells, but see §4f — the full sweep later showed **1,324 cases >5% slower** behind
that flat aggregate. An aggregate is not evidence about a distribution.

### 9. A mask was written 16-bit and read 32-bit

*What it is.* When the width is not a multiple of 32, the last tile is partly padding. The op
multiplies by a tile of 0s and 1s to zero the padding before summing.

*Wrong.* The mask buffer was allocated at bfloat16 **unconditionally**, while the reduce helper
reconfigures both of its unpack operands from the reduce's *input* buffer — so the mask is read
in whatever format the intermediates are. At 16-bit intermediates the two agreed by coincidence.
At float32 they did not: the lane pitch halves, the 1s land in the wrong places, some real data
is zeroed and some padding is counted. Fixed by making the mask follow the intermediate format.

*Where the 16-bit came from — traced 2026-09-09.* **Not** authored during this run. The seed
op hardcodes it, comment and all (`rms_norm_program_descriptor.py:1812`, `:2255`), and the
successor's first commit reproduces all three lines byte-identically under the preserve-and-extend
rule. Behind that, `agents/incremental-planner.md:228` carries an unconditional design-checklist
item — "Reduce scaler CB uses bfloat16 packed format" — with no carve-out for the mask case and
no mention that the format must match what the reduce reads.

And behind *that*, the only justification that ever existed is a **precision argument about the
wrong property**: the seed's design note argues that 1.0 is bit-exact in bfloat16, so a bfloat16
scaler is safe. True, and about the scaler's *value*. It silently licensed the *format* for a
mask, whose correctness depends on lane pitch rather than mantissa bits.

Two corrections to earlier readings of this: the `/partial-scaler-reduce` skill does state the
format is "independent of the input dtype; bfloat16 is the usual choice", which is false for
exactly this case — but it was **never opened** during the run (zero skill invocations across 22
transcripts), so it did not cause this. Fix it anyway; it will cause the next one. And the
intermediate promotion of fix 8 did **not** make this reachable: float32 builds are byte-identical
before and after, so the bug was live from the op's first commit. **The seed still has it today**
— float32 in SUPPORTED, the accumulate-via-add path at four or more width tiles, and the squared
buffer at the input dtype. Not back-ported.

*Why the tests missed it.* The worst case of the ten. The cartesian **ran** float32 non-aligned
cases at the right width. They passed because a mis-shifted mask produces a near-**uniform scale**
error, the correlation check is blind to uniform scale by construction, and the relative-RMS
tolerance at float32 absorbed the rest. The suite contains a group built *precisely* to cover
that blindness — padding filled with a poison value so a leak screams, six narrow non-aligned
widths, all four placements, with a comment saying it exists because the correlation check cannot
see this class of error. **It is pinned to bfloat16**, via `**_RESILIENCE_BASE`
(`feature_spec.py:573`) spread into each case sixty lines below, and bfloat16 is the one format
where the written and read formats already agree. Right shape, right detector, never crossed —
and the pinning is documented as a cost decision ("so each shape costs 4 cells, one per
placement") without noting what it gives up.

### 10. A block size the op approved, then crashed on

*What it is.* `subblock_w` is a caller-settable number: how many 32-wide tiles to process per
step in one compute stage.

*Wrong.* Validation checks it divides `block_w`, the width of one core's slice in tiles. Deeper
in, the compute loop walks a different quantity — the op's own resolved chunk width — and the
caller's number was never checked against that. Concretely, at width 72 over 2 cores: the row is
3 tiles wide, each core's slice is 2, a caller passing 2 divides the slice cleanly, the plan
walks 3, and an internal `assert` fired. A Python crash, on a call the op had just declared
valid. Fixed by taking the largest divisor of the resolved chunk not exceeding the request.

*Why the tests missed it — four layers.* `subblock_w` is not a swept axis, only a per-case
override. Every case that does build a tuning object uses a tile-aligned width, so the fallback
plan never triggers. The one non-aligned case that passes a tuning object uses the interleaved
variant, which has no `subblock_w` field. And the helper that derives a sharded tuning object
sets it to 1, which divides everything — so even adding a non-aligned sharded case would need an
explicit override above 1 as well. All four had to line up.

*Measured 2026-09-09, and it changes the reading.* This knob sizes **zero** L1 — see §4h. And
tracing the failing case with `RMS_TRACE_BLOCKING` gives
`scheme=rows cores=1 wt_per_core=3 WT_CHUNK=3`: **the op collapses to ONE core doing all three
tiles.** The shard genuinely spans two cores (core 0 holds tiles 0-1, core 1 holds tile 2 plus a
pure-padding tile — the test's own helper asserts the last core keeps at least one real tile),
and the op ignores that split and reads the neighbour's shard over the network.

So the test's purpose is defeated while it passes: it exists to verify a width split across
cores with a ragged last shard, and says so in its docstring, and the op gets the right answer
by a route the test was not written to exercise. Still open: whether the two-core split is
genuinely inexpressible in the op's cross-core scheme or the solver simply bails — that decides
whether this is a missing capability or a second bug.

*Contract note.* "HONOURED, never clamped, never absorbed" appears in the prompt, the manifest
item and the op's own comment, and is not achievable on plans that re-chunk the width. Since the
knob buys no memory and the op's derived default measured **better** than the hand-tuned value
(24,497 ns against 25,513), quietly taking the nearest legal value is defensible. Fix the
sentence, not the clamp — and delete the now-unreachable assert two lines below it.

### The four that were our tooling, not the op

* **A typo in our own test helper.** It read `.start_coord` / `.end_coord` off a core-range
  object. Those are the C++ member names; Python exposes `.start` / `.end`. 7 cases.
* **Zero-element results crashed the accuracy check.** Some shapes have no elements at all
  (`0x64`). The op handled them correctly, but the shared accuracy code asked torch for "the
  largest difference" across an empty set, which raises rather than returning anything. Three
  cases failed on shapes the op got right. (Fixed in the submodule — it was shared by every op.)
* **Four target test files were listed as "already present".** The reference-test system keeps a
  manifest saying, per upstream file, whether to restore it from a pinned commit or take it from
  the working tree. Four were marked "take from the tree", on the theory that the nuke deletes
  only files whose name carries the op's name. It actually deletes whole ops — and `layer_norm`
  is also a nuked op — so those four were gone. That wrong assumption is §4d's corrected
  paragraph, and it is what crashed the whole reference pass.
* **The file assembler wrote one file at a time.** It hit the first missing file and died
  half-way, leaving three of seven on disk. The graded run then collected those three as though
  they were our own tests and folded them into the headline score. It now resolves all seven
  first and writes only if every one is available.

### What to change in the prompt and the feature spec

Ordered by what it cost. Each is stated so it can be applied to an op nobody here has seen.

1. **Describe a constraint as a rule, never as one example that satisfies it.** Writing a shape
   as `(Wt, 32)` pins the number of dimensions even when the source rule does not mention rank.
   State the property the target's code actually tests. *(done — skill `4423661`)*
2. **Never hand-enumerate a combinatorial set.** N optional inputs means 2^N combinations; say
   that, not a list. A list is a test-budget decision masquerading as a capability decision.
   *(done — `510bc66`)*
3. **A note about a target quirk must say what a CALL DOES, not what to avoid.** "Do not silently
   accept a value you discard" is satisfied by refusing the call, which is a different contract.
   *(done — `abf8967`, `call_outcome`)*
4. **A golden tolerance must be at least as tight as the target's own for the same case.** Ours
   was ~6x looser on `bfloat8_b`, which makes a real regression undetectable by construction. Any
   gate looser than the target's is a hole with a number in it.
5. **A detector built to cover a specific blindness must be crossed with every format where that
   blindness can occur.** The padding-poison group exists because the correlation check cannot see
   uniform scale errors — and it is pinned to the one format where the bug it was built for
   cannot happen. Where a group is pinned for cost, write down what the pin gives up, not just
   what it saves.
6. **A property named in a shard spec is an axis.** Orientation cost 73 cases and is one enum
   field. If the target's own type carries it, the suite has to sweep it.
7. **Test a core grid larger than the work.** The "spare core" geometry never appeared and
   crashed on contact.
8. **A caller-settable knob reachable only through a per-case override is untested in practice.**
   If it is part of the public surface, sweep it — or accept that its behaviour is unknown.
9. **The test suite must never supply a missing piece of the op.** Concretely: if writing a case
   requires inventing a class, a constant, or a conversion the op does not provide, stop — that
   is a missing piece of the op's public surface, not a test fixture. A suite that fills the gap
   proves only that the op works when the harness completes it, which no caller will do. This is
   the single rule that would have caught defect 2, and my own defect 3 is what happens when the
   stand-in itself is subtly wrong.
10. **Re-examine inherited code against the axes a successor newly supports.** A seeded generation
    keeps the seed's decisions on purpose. Defect 9 was a 16-bit assumption that was correct for
    every format the seed shipped and wrong for one the successor added. Nothing in the pipeline
    looks for that.

Items 1-3 are landed. 4, 5, 7, 8 and 9 belong in `/golden-tests`; 6 in this op's spec and in the
general rule about shard properties; 10 is a seeded-generation rule with no home yet.

## 4j. Measured against three baselines, and one retraction (2026-09-09/10)

Report page (8 versions, five tabs):
https://claude.ai/code/artifact/619a6c74-bc2c-40a5-90ad-97aefc60af8f

Four comparisons, all on Blackhole p150b, all per-case `device_kernel_ns` through
`eval_test_runner.sh`. Every one is a NEW measurement, not a re-reading of the run.

### vs the target, on the target's own tests — 416 cases

All 416 paired, **all 416 faster**. Best 91.55x, median 2.68x, worst 1.04x. The top end is
almost certainly native running single-core where this op uses the grid — inherited from the
seed's architecture, not earned. The narrow margins are small row-major and sharded-residual
shapes at a few microseconds.

*Method note worth keeping:* the pinned upstream ref IS `tt-metal2`'s HEAD, so both sides ran
byte-identical test files. Pairing needed id normalisation — the two trees' pytest versions
render parametrize ids differently (`[3072-8192]` vs `[w=3072-h=65536]`, `dtype0` vs
`torch.bfloat16`). A first attempt lost 127 of 416 to that and, because the curve is sorted
best-first, the drop was invisible and flattered the median (3.07x vs the true 2.68x).

### vs the target, on production traces — 140 cases

The model-traced corpus (PR #127): 140 distinct rms_norm calls from 22 production models,
334,159 recorded executions, distilled from the `ttnn_ops_v6` trace DB. **No translation was
needed** — the op accepts the blocked weight form, native's compute-config object and native's
`weight` kwarg, so all 140 replay in native's own dialect with only the op swapped.

140/140 pass both sides. **138 faster**, 2 ties (0.991x, 0.994x). Median 1.219x, geomean
1.506x, execution-weighted 2.157x. The margin narrows with size — <1 MB median 1.475x,
>100 MB median 1.018x — so more than half the corpus being small is why the weighted figure is
high. **The corpus contains no bias and no residual anywhere**, so two of the three optional
operands never appear in real traces.

### golden before/after — the cost of the ten fixes

23,458 cases timed at both `fa3a545553` and `13246e1767`. **Zero pass -> fail.** 310 fail ->
pass. PCC: 3,672 better, 90 worse (all ~3e-06 at 0.99994, noise).

But **1,324 cases >5% slower**, worst 0.519x, behind an aggregate of 0.9965x. The implementer's
own 10-cell sample reported "net flat" and was right about the aggregate and blind to the
distribution. Two causes, and the big one is not the one anyone guessed:

* **1,152 (87%) are `FLOAT32 x w_non_aligned`, one-sided** (1,152 slower, 1 faster). Cause is
  the mask-format fix: `scaler_dtype = interm_dtype if kernel_partial_w else bfloat16`
  doubles that tile at fp32, and the reader zero-fills it behind a BLOCKING barrier at boot.
  Fixed additive cost ~+200-325 ns, so the percentage loss scales inversely with runtime —
  +325 ns on 2,126 ns is 0.873x, +58 ns on 9,226 ns is 0.994x. The worst shape, `32x17`, is a
  SINGLE TILE, which rules out any re-blocking explanation. ATTEMPTS priced this at zero
  ("the only float32 movement is +4 kB of fixed scaler CB") — that parenthetical is the 87%.
  **Recoverable:** the larger mask is only needed on the accumulate-via-add path (>= 4 width
  tiles), but the gate is on partial-width; every worst shape is 1-2 tiles wide, on the other
  path, which the commit itself says is correct at either format.
* **113 are `BFLOAT8_B x HEIGHT_SHARDED`** — the intermediate promotion, genuinely bimodal
  (113 slower / 107 faster, median 1.0006), a re-priced L1 solve reshuffling the blocking.
* The residual 58 are noise (symmetric, median ratio 1.0000).

**Zero core-count changes in the entire run** (0 of 23,458), so no regression is a work-split
story, and shard orientation is excluded outright.

### vs the seed, on the SEED's own suite — 5,327 cases

The requirement-6 test. Adapter binds either op, asserts the seed LACKS
`weight`/`bias`/`residual_input_tensor` and the successor HAS all three, records source sha256
per side, and the joiner refuses if both resolve to the same callable. `gamma_mode` restricted
to `{no_gamma, gamma}`; bias/residual never in the kwargs dict at all. Both ops run in ONE
clone (it carries a byte-identical copy of the seed), so one build, one `_ttnn.so`, one device.
Two replicates in opposite order, min-of-2, agreeing to 0.23% median.

| slice | n | geomean | seed faster >5% |
|---|---|---|---|
| all paired | 5327 | **1.188x** | 81 |
| **weight ABSENT** | 986 | **1.141x** | **2** |
| interleaved / height / width / block | | 1.078 / 1.186 / 1.240 / 1.285 | |

**Requirement 6 is not satisfied as timing identity.** Core counts match on all 5,327, so the
work split IS the seed's — but the operand-free path runs ~14% faster, so the code is not. The
guard test (`test_program_is_structurally_the_seeds`, 72 cells) compares the CB set and kernel
args, **not kernel source**, and the perf rounds rewrote the kernel. Requirement 6 says "same
buffers, same code path, same blocking"; buffers and blocking are checked, code path is not.

The deviation is benign in direction — nothing leaked cost INTO the base case — but "equivalent
to the seed's program" is not what shipped.

### RETRACTED: the 71-cell W=8192 regression

This run also reported 49 cells the seed passed and the successor failed, plus 22 skips, all at
W=8192, reproduced byte-identically in both replicates. **None of it is real.** Recorded here
because it cost a day and because the reasoning error is the reusable part.

The failure message carries two numbers: the arena's end, and `L1 buffer allocated at`. The
second is not a property of the op — it is whatever was still live in that pytest session. It
takes 6 values on one side and 8 on the other **with zero overlap**, and 68 of the 70 cells
would fail on BOTH ops at the other's threshold.

Cause: a cell that RAISES leaves pytest holding its traceback, whose frames still reference its
on-device tensors, in a gc cycle refcounting cannot break. The pinned L1 collides with the next
cell's CB region — 1,247 consecutive affected cells in this trace. The hook that releases them
(`eval/golden_tests/conftest.py:217-218`) is real and correct, and **never loaded**: pytest
auto-loads that conftest only for tests UNDER `eval/golden_tests/`, and the runner injects only
hang/metrics/axes by `-p`. Force-loaded, **both ops land on exactly 5421/0/21** and residue
goes from 1,253 of 5,442 cells to zero. Filed as **tt_ops_code_gen#193** with the repro and two
fix directions; also caught `eval/oom.py` charging 24 of 45 `INFEASIBLE_L1` skips to shard
geometry when they were residue.

**The reasoning error, which is the point:** I treated reproducibility as intrinsicness. Same
result across two replicates, both orderings, eight phases — none of that distinguishes a
property of the op from a property of the environment it was measured in, because the
environment repeated too. The cheap discriminator I should have reached for first is
ISOLATION: the cells pass one-per-process. A read-only probe relocating the entire failing set
(9 of 49 in common) would have settled it in minutes.

Three of this session's conclusions were overturned by looking one level deeper — the skill
blamed for the mask format (never loaded), the intermediate promotion blamed for making it
reachable (fp32 byte-identical), and this. All three were the same error.

### The perf rounds: which shapes, and the `attention:` bug

Perf case designation is `extras["achievable_ns"]` + `reference_aiclk_mhz`, NOT the
`"group": "perf"` label (which the spec says never affects gating, `feature_spec.py:236-241`).
Selection is a written rule since submodule `0a8a442`: measure every perf case, divide measured
ns by that case's own achievable, take the largest ratio, **re-rank every round**.

Before that, the prompt made a `# attention:` note the mandatory target. It was never a tag or
a group — a bare Python COMMENT in one LOOSE_CASES entry, appearing exactly once in the tree,
parsed by nothing. Where present (sdpa) one shape was mandatory for every round with no
re-rank; where absent (rms_norm run 893) the fall-through had no procedure, so the agent
improvised and recorded three times "no `attention:` note, so the focus shape was
free-selected", landing on the same shape all three rounds. `4d5dbe1` then widened the ranked
population 13 -> 19.

**It worked.** Run 998's focus was a different shape AND placement each round (width-sharded
decode -> two different interleaved prefills), 18 of 19 cases ended faster, zero >5%
regressions, and one of the six cases the widening added was Perf 2's largest prize at 1.49x.

Two caveats. The ranking is dominated by REFERENCE QUALITY, not headroom — ratios span 0.06 to
0.997, so a case whose target is 12x off can never be picked however much time it wastes
(issue #175). And the focus stopped being the mover: Perf 2's focus moved 4.1% while the
round's value landed on two cases it was not targeting; Perf 3's moved 0.3% while ten other
cells moved 5-15%. The changelog reports this honestly. The honest metric is the worst-ratio
walk, 0.997 -> 0.908 -> 0.871 -> 0.857.

### What these four measurements add to §4i's list

11. **An aggregate is not evidence about a distribution.** "Net flat over 10 cells" concealed
    1,324 regressions with a worst case of 0.519x. A perf claim needs the count above and below
    the threshold, not a mean.
12. **Reproducibility is not intrinsicness.** Before attributing a failure to an op, run the
    cell in isolation. It is one command and it would have saved a day here.
13. **A sorted chart must show what it drops.** 127 of 416 cases silently unpaired, and 23,458
    bars drawn 1px apart on a 1,000px canvas so every regression fell off the right edge —
    both flattered the result in the same direction.

## 5. Playground: the op and its gap

Everything below is `rms_norm`-specific. It is evidence, not requirement.

### Tree state

| | |
|---|---|
| Branch | `dnijemcevic/agent_eval_new`, HEAD `09305879fec` — *"rms_norm: port the agent-generated op onto this baseline"* |
| Submodule pin | `b4294a3` on `dnijemcevic/codegen_new` (was `14d2784` at the start of 2026-09-03) |
| Pristine main | `/localdev/dnijemcevic/tt-metal2`, built with the profiler — where the native op can actually be measured |
| Seed op | `ttnn/ttnn/operations/rms_norm/` — `op_design.md`, `changelog.md`, `op_requirements.md`, `verification_report.md`, `kernels/`, descriptor |
| Seed lacks | `l1_ledger.md`, `agent_logs/`, `perf_experiments/` |
| Golden suite | `.claude/eval/golden_tests/rms_norm/` — six-file layout + `test_translated.py` + `translate_stats.json` |
| **Target** | **NUKED locally** — read from `origin/main` via `git show` |

Original run branch (kernels, perf experiments, agent logs): `dnijemcevic/2026_08_04_1241_rms_norm`,
tip `bfe3e673a6a`.

### Settled for this instance

| Question | Decision |
|---|---|
| Staging name | `rms_norm_ttnn` |
| Target source | `origin/main` via git — works regardless of local nuke state |
| Golden suite | Copy-and-extend the seed's (preserves tolerances, helpers, `axes.py`; mechanises 20) |
| `test_traced.py` | Out of scope — lives in an unmerged PR |

### The gap

13 differences; 2 already match, 1 is a case where the generated op is *wider* than the target
(harmless — no caller can depend on a rejection). **10 to close.**

**Surface — no blocking impact:** scale tensor named `weight` vs `gamma`; `epsilon` default
`1e-12` vs `1e-6`; `program_config` accepted but silently ignored; compute-config object is a
different type; input rank 0/1/5+ accepted by the target, 2–4 by the seed; no registered golden
function (so the framework's lookup breaks); a public default-config helper not provided.

**New optional operands — these are what 6–9 govern:** `bias` (per-column, same shape and
broadcast as the scale tensor), `residual_input_tensor` (full-size second input), and `inplace`
(carried inside `program_config`; output aliases input, changing the contract about input
survival).

**The math — both implementations agree on all of it:**

```
1.  x' = x + residual      (optional — added BEFORE statistics; a real fusion,
                            not expressible as "normalise then add")
2.  m  = mean(x'^2) along the last dim
3.  r  = sqrt(m + epsilon)
4.  y  = x' / r
5.  y  = y * weight        (optional — per-column, broadcast down rows)
6.  y  = y + bias          (optional — same shape and broadcast as weight)
```

Every difference is naming, shape, or presence — never meaning.

**Coverage:** ~70 source lines expanding to ~600 live test cases across 13 files. `inplace` ~290,
compute-config ~420, residual ~186, bias ~153, relying on the old `epsilon` default ~204. Two
config fields have **zero** cases and are inert for this op — accept and ignore them, as the
target does.

---

## 6. Playground: facts worth not re-deriving

### Which artifacts are ground truth (evidence for 17)

Verified across all prompts and agent definitions:

- **`op_design.md`: write-once, and provably stale.** For the seed it contains **zero** mentions
  of the third compute regime, of "Refinement", or of any placement scheme — while mentioning
  "Phase 0" 31 times. It pins a two-way regime branch and asserts an identity the shipped op is
  built on *breaking*. 10 buffers designed vs 19 shipped; one placement scheme vs four. Five of
  its six deferred items have since shipped.
- **`verification_report.md`: write-once**, and likewise stale on at least one axis.
- **`changelog.md`: maintained** every phase.
- **`op_requirements.md`: maintained for generality refinements only** — perf rounds leave no
  trace in it.
- **`l1_ledger.md`: absent from every op in the tree**, despite being a planner deliverable with
  a keep-current mandate and an ingest hook.
- **The genuinely current record is the numbered design notes in the descriptor's module
  docstring** (D1…D28 for the seed) — current because it annotates retirements in place
  ("RESOLVED by …", "PERF 1 RETIRED …", "D22 refutes D16's recorded reasoning"). **Caveat: an
  emergent convention of this op, mandated nowhere.** Do not assume the next seed has one.

**Conclusion:** a derived code-analysis pass is required, not optional — hence `/analyze-op`.
The maintained artifacts are chronological, not structural; a reader would have to reconcile
eleven phases and their retirements to recover the shipped design.

### The seed's blocking

Three regimes chosen by a host-side solver (`rms_norm_program_descriptor.py:1883`, inside
`create_program_descriptor` at `:1778`). **One kernel** — the regime arrives as compile-time args.

| Regime | input resident | width chunks | memory reads/row |
|---|---|---|---|
| RESIDENT | yes | 1 | 1 |
| ROW_RESIDENT | yes | > 1 | 1 |
| STREAM | no | > 1 | 2 |

- `x_hold_wt = wt_per_core if x_resident else wt_chunk` (`:2142`) — the whole regime, one line
- `static_assert(!ROW_RESIDENT || BLOCK_ROWS == 1)`; `static_assert(!CROSS_CORE || NUM_W_CHUNKS == 1)`
- Large interleaved shapes all land at `BLOCK_ROWS = 1`; multi-row blocks only when the width is
  already split across cores
- Held full-width: input tiles, scale tiles. Chunked: squares, normalised, output
- The row-statistic buffer has **no width term** — the reduce collapses width, which is what makes
  chunking work. Transformed in place from sum-of-squares to reciprocal-RMS
- Buffer depth 2 exists because pass A of block *n+1* overlaps pass B of block *n*

Adding operands shifts the regime thresholds — measured for this op: single-read survives to
~10,048 values/row today, ~6,720 with bias, ~5,024 with both. One production shape sits at 96.4%
of budget, and the regime transition was measured at 1.6×. **This is per call configuration**
(requirement 8): a call omitting the operands keeps today's blocking.

### Optionality — how the target does it (evidence for 6–9)

**Device side is genuinely zero-cost**, with two techniques worth copying:

- the *destination* buffer is chosen by configuration, so a weightless call writes straight to
  output with no staging hop
- one configuration aliases an intermediate onto the input buffer, dropping a full-width buffer

No identity tensor is materialised anywhere; `inplace` never reaches a kernel.

**Two leaks, both host-side, and both are why 7 and 8 are stated separately:**

- **Requirement 7 violated:** the fit predicate charges for absent optional tensors — their
  *format* defaults when the tensor is `nullopt`, so a nonzero tile size flows into the budget
  sum. A no-operand call is charged ~2.5× what it allocates and takes a fallback path ~2.6×
  earlier than it needs to. Instructively, one of the three optional inputs *is* handled
  correctly in the same function — the property was asserted per operand and missed twice, which
  is exactly why 19 makes it a mechanical check.
- **Requirement 8 violated:** the fallback path's block-size caps come from a two-bucket
  constant ("with weights" / "without"), sized for the worst-case operand set. A call using one
  optional input gets blocks sized for three — half-size blocks, twice the iterations. The
  source comment admits it: *"having two constants for all cases is simpler."*

**Variant count is not the constraint you would expect.** The kernel hash includes the full
compile-time-arg vector, which for this op family includes the tensor width — so *every distinct
width is already a separate binary*. Operand presence adds a small multiplier on top of an
already-unbounded axis. Separately, three defines are emitted to kernels that never reference
them, multiplying the cache for nothing.

**A live typo argues for compile-time args over preprocessor defines:** one guard in the target
misspells a define name, so a negative-form condition is unconditionally true and fires in the
case it was written to exclude. Three sibling guards spell it correctly. A `#define` guard has no
compiler safety net; a compile-time arg with `if constexpr` does.

### Memory adaptivity — neither implementation queries free memory

Both plan against an architectural constant. **But 14 call sites elsewhere in the repo query
actual remaining L1**, including a sibling normalisation op, which also pre-accounts for a buffer
allocated after planning. So this is an established in-repo pattern that neither of these two
adopted — not a framework limitation.

Python exposure is half-present: the base-address query is bound, the lowest-occupied query is
not. Closing that is one binding, not a port.

### Mechanism facts

- Both implementations return the same program-descriptor type; the difference is host language.
  One inventory tool therefore covers both, letting a gap spec carry a *measured* footprint
  comparison rather than only a signature comparison.
- Descriptor-built programs **are** cached on a structural, address-independent key.
- The decode path **traces** this op, so host-side planning runs at capture, never at replay —
  which is why requirement 3 can put a language port out of scope.
- The Python gap is the fast cache-hit path (no buffer bindings exposed), affecting untraced
  calls only.

### Prohibitions that block a seed (evidence for F1)

The integrity text's categorical bullet forbids reading any pre-existing implementation of the op
*"whether in the working tree or in git history"*. That catches a designated seed.

- The temporal baseline rules do **not** block it — they govern history *retrieval*, and a seed
  sits in the working tree.
- The planner's former "do not mine a prior attempt" rule is **gone** from the current pin, so
  only one prohibition remains.
- The mismatch: the baseline mechanism encodes *time*; designation is about *declaration* (14).
- **Fix:** the categorical bullet defers to an explicit designation in the op's prompt.

### Eval-system findings

- **Reference-test discovery undercounts by ~3×** for this op — it recorded 192 cases where the
  real surface is ~600 across 13 files, because discovery keys on the op's name and most of its
  coverage lives in files named after a sibling op. Prerequisite for a sound delta.
- **The target's registered golden function is already wrong** — it silently discards two of the
  optional inputs. A bug to fix, not a behaviour to reproduce.
- The seed's `op_design.md` has 9 known divergences from its code.

---

## 7. Open questions

- Does a mock device suffice for `l1_inventory.py`, or does it need real hardware?
- How many gap→generate→re-audit iterations before convergence?
- Does a successor inherit the seed's tuning work, or re-derive it? The seed's design notes are
  readable by the planner; whether that transfers in practice is untested.
- ~~F3's wording~~ — answered 2026-09-04: 6 and 16 partition rather than conflict, so the rule is
  "design the second without disturbing the first". See §4c.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Successor regresses cases the seed passed | F4, mandatory (5, 20) |
| Planner diverges gratuitously instead of extending | The `## Rules` partition (landed); F5's build-with-and-without check makes the operand-free half mechanical, not advisory; `/analyze-op` output as the diff baseline |
| Absent-operand paths acquire cost | F5 mechanical check (19); the target violates this twice today |
| Delta incomplete → successor still does not bridge | Discovery fix + the ratification gate (18) |
| Seed designation abused | Prohibition stays default-deny; the prompt names the one permitted seed (14) |
| Adding operands silently costs performance | Mode C's phase-3 report surfaces it before the run |

## 9. Deferred

- Adaptive memory budgeting (one binding + swap the budget source; a sibling op is the reference)
- A host-language port — optimisation only, and requirement 3 puts it out of scope
- Keeping both a ledger and a measured inventory: ledger = intent, inventory = truth, verifier
  flags disagreement
- Root fixes outside this flow: either maintain the design doc through refinements or stop
  handing it downstream as "the design"; and make the pipeline notice when a mandated artifact
  was never written
