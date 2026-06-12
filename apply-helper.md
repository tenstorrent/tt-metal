---
name: apply-helper
description: Roll a signed-off kernel-helper proposal out across the codebase. Takes a design proposal package (proposed_helpers.md + census.txt + migration_audit/* from design-helper or tune-helper), MATERIALIZES the helper (consolidates the bake-off winners into the real kernel_lib helper + a unit test ported from the bake-off harness), builds a DEVICE-VERIFIED kernel→test map (kernel → program factory + dispatch condition → op → the exact test params that hit it, proven by JIT build-cache or a compile-time marker — not static grep), sorts the call sites into ordered migration TIERS (easiest/safest first by audit tag × API distance × verified-coverage strength), then migrates them one kernel at a time, each guarded by its own mapped tests. Phase 4 runs ONE SUBAGENT PER TIER (strictly sequential — shared device + JIT build) so the orchestrator's context stays lean: each subagent migrates its tier in an isolated context, writes the verbose diffs/logs to disk, and returns only compact conclusions. INTERACTIVE at two human gates (intake, plan); autonomous within a tier. The failure mode is REQUIRED from the invocation prompt — `--mode=halt` (stop at first failure, leave the diff to debug) or `--mode=run-all` (revert each failure to keep the tree green, one report). Per-kernel atomic git commits make revert + bisect clean. Ends in a migration report. Use when you have a signed-off helper proposal and want to apply it to a FLEET of call sites with a safety net — not for one ad-hoc kernel. Args = <proposal-dir> --mode=halt|run-all [--tiers=...].
---

# Apply a Kernel Helper Across the Codebase

Use this skill when you have a **signed-off helper proposal** (1–2 fat helpers, baked-in style
choices, knobs, internal dual-paths) plus a census/migration-audit, and you want to **roll the
helper out across every call site** with a safety net — materializing the helper, knowing exactly
which tests guard each kernel, and migrating in a controlled, reportable way.

It is the **rollout successor** to the design skills. `design-helper` / `tune-helper` *exit at a
proposal*; `apply-helper` *consumes* that proposal and owns everything from "materialize the helper"
through "migration report."

```
design-helper / tune-helper   produce proposed_helpers.md (the design)         — exits at proposal
apply-helper                  materialize → map → tier → migrate → report      — does the rollout
```

## When to use
- You have a signed-off `proposed_helpers.md` + `census.txt` + `migration_audit/*` and want to apply
  the helper to a **fleet** of call sites.

## When NOT to use
- The helper isn't designed yet → run `design-helper` / `tune-helper` first.
- You're migrating one kernel ad hoc → just do it; this skill is for fleets.

---

## Interactive checkpoints — read this first

**This skill is interactive at two human gates, autonomous in between.** Stop and get an explicit OK
at **Gate 0** (intake) and **Gate A** (plan + mode). Inside a tier, the per-tier subagent runs
autonomously per the chosen `--mode`. After every gate:

```
1. Hand the user the artifact(s) just produced.
2. Summarize in ~5–10 lines what's in them and the load-bearing facts.
3. End with a clear question and WAIT for an explicit OK before proceeding.
```

Gate A is a **Gate-2-equivalent**: it precedes touching any production op, so the user must accept
the map coverage (incl. the gap list), the tiers, and the failure mode before migration starts.

---

## Invocation

```
apply-helper <proposal-dir> --mode=halt|run-all [--tiers=1,2] [--helper-path=...]
```

- `<proposal-dir>` — e.g. `helper_design/mcast_pipe/` (the proposal package, see Inputs).
- **`--mode` is REQUIRED and read from the prompt — there is NO baked default.** The caller decides
  per run: `halt` (stop at first failure, leave the diff to debug) or `run-all` (revert each failure,
  sweep everything, one report). If the prompt omits `--mode`, ask for it at Gate 0 — do not assume.
- `--tiers` (optional) — restrict the run to specific tiers (e.g. just the clean spine).
- `--helper-path` (optional) — where to materialize the helper (default: a sensible `kernel_lib` path).

---

## Inputs (Gate 0)

**Required** (the proposal package):
- `proposed_helpers.md` — helper API, baked-in style choices, knobs, internal dual-paths, and the
  **migration list with refactor costs** (clean/refactor/defer tags).
- `census.txt` — durable manifest of every call site, grouped by op-family + tagged.
- `migration_audit/*` — per-group clean/refactor/defer inventory + headline blockers.

**Strongly recommended:**
- `style_bakeoff.md` + the `bakeoff_*` kernels — the **proven raw baseline**; the bake-off kernels
  ARE the winning variants, so Phase 1 consolidates them rather than inventing.
- `hazards_catalog.md` — so a migration that trips a hazard is recognized, not rediscovered.

### Checkpoint — Gate 0 (intake)
Confirm the proposal package is present and readable; echo back the helper name, the count of
clean/refactor/defer call sites, and the **`--mode`** the run will use (ask if absent). Wait for OK.

---

## The pipeline

```
Gate 0  — Intake            confirm the proposal package + the --mode arg
Phase 1 — Materialize       consolidate the bake-off winners into the real helper + unit test  [OWNED]
Phase 2 — Kernel→test MAP   for every census kernel, find AND device-verify which tests exercise it
Phase 3 — Tier the rollout  group call sites into ordered migration tiers (easiest/safest first)
Gate A  — Plan sign-off     present map coverage + gaps + tiers (+ echo the chosen mode); OK to proceed
Phase 4 — Migrate loop      ONE SUBAGENT PER TIER (sequential); subagent migrates its kernels,
                            returns only conclusions; orchestrator keeps a lean context
Phase 5 — Report            per-kernel + per-tier results, coverage gaps, deferred, perf deltas
```

Human gates: **Gate 0** and **Gate A**. Everything else is autonomous within the chosen `--mode`.

### Context discipline — per-tier subagents (orchestrator stays lean)

Migration is verbose (per-kernel diffs, build logs, pytest output, triage JSON). If the orchestrator
did it inline, its context would explode across dozens of kernels. So **Phase 4 runs one subagent
per tier**:

- The orchestrator spawns **one subagent for tier N, waits for it, then spawns tier N+1.** Tiers run
  **strictly sequentially** — never in parallel — because they share the device + JIT build, and only
  one device test may run at a time. (Kernels *within* a tier are also sequential, inside that
  subagent.) **Subagents isolate context, not execution.**
- Each subagent gets a **fresh, isolated context**. The orchestrator passes only **pointers + the
  tier worklist**: the helper path, `proposed_helpers.md`, the kernel's audit notes, the
  `test_map.json` entries for *that tier's* kernels, the `--mode`, and the migration conventions
  (atomic per-kernel commit, `--dev`, revert-on-fail). The subagent reads detail from disk itself.
- The subagent **writes all verbose detail to disk** (`migration/log/<kernel>.md`, its slice of
  `migration/report.md`) and **returns only the conclusions** — a compact structured summary, never
  the diffs / full test output.

**Subagent return contract (keep it small):**
```
tier: N
per_kernel:
  - kernel, status (migrated|failed|deferred), validation_result (pass|fail|hang),
    commit_hash (if migrated), diff_lines_removed, coverage_gap? (y/n), one-line note
tier_totals: {migrated, failed, deferred}
halted_at: <kernel or null>     # set only in halt mode when a failure stopped the tier
artifacts: [paths to the log files it wrote]   # pointers, not contents
```

---

### Phase 1 — Materialize the helper (OWNED by this skill)

Migration to a non-existent helper is meaningless, so this skill builds it:

1. Consolidate the baked-in winners + internal dual-paths from `proposed_helpers.md` into the real
   `kernel_lib` helper (e.g. `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp`), on the same object API the
   bake-off used. The `bakeoff_*` kernels are the reference for each winning variant.
2. Port the bake-off harness (`test_<helper>.py`) to drive the **helper** instead of the raw
   `bakeoff_*` kernels — same coverage matrix, now the helper's unit test. Green = the helper
   reproduces the bake-off behavior.
3. Confirm the proposal's **provisional items** (e.g. a consumer-wait ordering assumption; an
   in-context perf claim) against one real kernel before mass rollout.

Artifact: the helper file + its unit test passing.

#### Materialization invariants (the helper's *implementation contract* — enforce these here)

The proposal decides *what* the helper does; Phase 1 decides *how it is built*. These invariants
are not negotiable at materialization and are cheap to violate by accident — check each one:

1. **Object/2.0 API only — no raw free functions, anywhere.** Every primitive call inside the helper
   goes through the object layer (`Noc`, `Semaphore<>`, the `*Endpoint` address types) — including
   the ones it's tempting to leave raw because "the object API doesn't have an overload for it" (the
   multicast send, an open-coded `get_noc_multicast_addr`, a bare `noc_async_read` for a self-copy).
   If a needed overload is genuinely missing from the object API, that is a **gap to flag to the
   user**, not a license to drop to the legacy call. A helper whose docstring claims "built on the
   object API" while its body still calls `noc_async_write_multicast` is a materialization bug.
2. **Split asymmetric faces into separate types.** If a "two-sided" helper has faces that take
   *different* constructor inputs (e.g. a broadcaster needs a destination rectangle + recipient
   count; a listener needs neither), materialize them as **two types**, not one type with dead args
   the listener fills with "pass 1 by convention." Dead ctor args are a design smell that survived
   into the implementation — kill them. Per-call context the listener *does* need (e.g. the sender's
   coords for its ack) becomes a **method argument**, not a ctor field.
3. **Caller-facing counts must be statable without knowing the internal dispatch.** A count the
   caller passes should be derivable from the caller's own topology/config alone (e.g. "the full
   number of cores that receive the broadcast, including me if I'm one of them"). The helper then
   derives every mode-specific hardware count (mcast `num_dests`, ACK count, ±1 for loopback)
   *internally*. Never make the caller pre-subtract for a mode the helper is supposed to infer.
4. **The helper owns its synchronization-primitive lifecycle — but ONLY race-free inits.** Take
   primitive **IDs** and construct the `Semaphore<>` (or equivalent) inside the ctor. A cell may be
   **kernel-initialized in the ctor only if this core establishes a happens-before edge to every other
   writer of that cell before they write it.** Otherwise the init races and the initial value must come
   from the **host** allocation, not the kernel. Worked example (cost: a multi-hour hang found at
   migration): a receiver initializing its own data-ready flag is safe (it writes the flag before its
   own ack, and the sender — the only other writer — is gated behind that ack). A sender initializing
   the ACK **counter** is NOT safe: receivers increment it remotely with no ordering vs. the sender's
   ctor, so a ctor `set(0)` clobbers an early ack and hangs — its 0 must come from host
   `CreateSemaphore(..., 0)`. Rule of thumb: **a cell that a *remote* core writes (a counter others
   increment, a flag another core mcasts into) cannot be ctor-initialized by the waiting side** unless
   the protocol provably orders that core after this ctor. Same-core-only-until-first-sync cells are
   fine. If the
   census shows call sites uniformly pre-seed a value the helper *writes* (not just waits on) — e.g. a
   broadcaster setting its local "ready" cell before the loop — fold that into the ctor too, as an arg
   defaulting to the **most-frequent observed value**, so the common call site drops the manual line
   and the rare one overrides via the arg.
5. **Don't materialize a knob no in-scope call site exercises.** Before building a proposed dual-path /
   template knob, check the census: if every `clean`/`refactor` call site takes the same arm and the
   other arm's only consumer is a `defer/raw` (out-of-scope) kernel, bake in the single arm and leave a
   note to re-introduce the fork as a refinement when that kernel is migrated. A knob with zero shipping
   callers is dead config that every real caller still has to read past. (Confirm by grepping the actual
   instantiations — e.g. all N `Pipe<>` uses defaulted `LINK`, so `LINK` was removed.)
6. **Make each arg compile-time (template) or runtime by what the kernels actually do.** For every value
   the helper takes, grep how the call sites source it. If it is `get_compile_time_arg_val` *and*
   identical across all cores running the binary (sem ids, a fixed recipient count, a literal init
   value), make it a **template / constexpr** param — the type then states "compile-time," the host
   can't pass a per-core value by accident, and addresses fold. Keep it **runtime** only when it truly
   varies: per-core under one binary (a per-row mcast rect → `get_arg_val`, set per-core), a CB
   read/write pointer, or a per-iteration / rotating value (a method arg, not a ctor field). Verify by
   reading a representative kernel — don't assume runtime just because the legacy code used `get_arg_val`.

### Checkpoint — after Phase 1
Report the helper path, that the ported unit test is green, and the result of each provisional-item
confirmation. Flag any provisional item that *failed* to confirm (it may force a proposal revision).
Wait for OK.

### Phase 2 — Kernel→test MAP (device-verified; the detail-oriented core)

For **every kernel in `census.txt`** build a precise, **device-verified** map of which tests exercise
it. Static grep is the START, not the answer — the goal is *verified* coverage. Resolve the chain
**kernel → program factory (+ dispatch condition) → op → the params that hit it**, then **prove it on
device**:

1. **Kernel → factory.** Grep the kernel filename across host C++ to find what instantiates it and
   under **what dispatch condition** (e.g. "matmul picks this reader only when 2D-sharded + mcast in0
   + `num_cores > 1`"). The condition is what separates "a matmul test" from "a matmul test that
   actually runs this kernel."
2. **Factory → op → tests.** Map to the TTNN op; enumerate its tests (sanity / nightly / sweeps),
   reusing the existing op-test-extraction approach where one exists.
3. **Narrow to hitting params.** From the dispatch condition, pick the specific parametrizations that
   compile-and-run this kernel — not the whole op suite.
4. **VERIFY ON DEVICE (required).** For each candidate case, prove the kernel actually ran — run it
   under `run_safe_pytest.sh` and check the **JIT/kernel build cache** for the kernel name (or a
   temporary compile-time marker / `DPRINT`). Record the method + result. A case that can't be
   verified to hit the kernel is marked `unverified` and does NOT count toward coverage.
5. **Classify.** Per kernel: `validation_set` (minimal fast verified cases to gate a migration),
   `regression_set` (broader), `coverage_confidence` (high/med/low), and **`gaps`** (kernels with
   weak / nightly-only / sweep-only / absent verified coverage). **Gaps are migration risks and a
   first-class output**, surfaced at Gate A.

Because this is a lot of kernels, **fan the mapping out — one subagent per op-family group** (as in
`tune-helper`'s census), each returning the map entries for its group; the orchestrator merges them.
Device-verification runs are sequential.

Artifact: `migration/test_map.md` (human) + `migration/test_map.json` (machine, consumed by Phase 4).
**Kept and reused** — the safety net for the rollout and a durable record for future audits.

> Example entry (sketch):
> ```
> kernel: .../reader_bmm_tile_layout_in0_sender_padding.cpp
> factory: matmul_op_multi_core_reuse_mcast_2d...  (selected when: 2D sharded, mcast in0)
> op: ttnn.matmul
> validation_set:
>   - tests/.../test_matmul.py::test_..[shard..-mcast..]  (sanity, fast, VERIFIED: kernel in build cache)
> regression_set: [ ...nightly sharded matmul cases... ]
> coverage_confidence: high
> gaps: none
> ```

### Phase 3 — Tier the rollout

Sort `clean`+`refactor` call sites into **ordered tiers**, easiest/safest first. Ranking inputs:
audit tag (clean < refactor-low < refactor-high), API distance (a trivial single-shape use vs a
multi-knob / multi-phase / dual-path use), and **verified coverage strength from Phase 2** (migrate
where you can validate cheaply first). Within a tier: ascending diff size × descending coverage
confidence.

Tiering template:
- **Tier 0 — proof:** the helper unit test (Phase 1) + one synthetic op. Already green.
- **Tier 1 — clean spine:** the `clean`-tagged, trivial-API, strongly-tested call sites.
- **Tier 2 — refactor-low.**
- **Tier 3 — refactor-high** (multi-phase / dual-path / hardest API distance).
- **Deferred (not migrated):** everything the audit tagged `defer` (out-of-scope shapes,
  legacy-API call sites that need a prerequisite port, etc.).

Artifact: `migration/tiers.md` — ordered worklist, per-kernel tier + rationale + `validation_set`.

### Gate A — Plan sign-off (human)

Present: (1) **map coverage summary** + the **gap list** (risky migrations with weak/no verified
coverage); (2) the **tiers** + order; (3) **echo the `--mode`** the run will use. Get an explicit OK
before any production op is touched. Wait for OK.

### Phase 4 — Migrate loop (per-tier subagent; autonomous within the chosen `--mode`)

The orchestrator runs the tier loop; the **per-kernel work happens inside the tier's subagent**:

```
# ORCHESTRATOR (lean context):
for tier in tiers:                      # strictly sequential — shared device + JIT build
    summary = run_subagent(tier, {helper, proposed_helpers.md, audit notes,
                                   test_map.json[tier kernels], --mode, conventions})
    append summary to migration/report.md ; surface to user
    if summary.halted_at is not null:   # halt mode hit a failure
        STOP (leave the failed diff in the tree)        # do not spawn the next tier

# SUBAGENT for one tier (its own context; kernels sequential):
for kernel in tier.worklist:
    snapshot (git checkpoint)
    rewrite the call site(s) to use the helper          # per proposed_helpers.md + audit notes
    run kernel.validation_set under run_safe_pytest.sh --dev   # --dev so a hang is captured
    if pass:
        (optional) run a slice of regression_set
        commit "apply <helper> to <kernel>"
        write migration/log/<kernel>.md ; record PASS
    else:  # fail or hang
        write migration/log/<kernel>.md (triage path, failing case, diff)
        switch (--mode):
          halt    -> STOP this tier, set halted_at=<kernel>, LEAVE the diff in the tree, return early
          run-all -> `git restore` this kernel (tree stays green), mark FAILED+quarantined, continue
    return only the compact conclusions (the return contract above) — never diffs / full logs
```

**`--mode` (from the prompt, required), end-to-end:**
- **`halt`** — the first failing/hanging kernel stops its tier subagent (diff left in place); the
  orchestrator then does **not** spawn the next tier. Debug from exactly there (optionally hand the
  failure to `debug-ttnn-op`).
- **`run-all`** — each subagent reverts its own failures (tree stays green) and finishes its tier; the
  orchestrator runs every tier and assembles one report.

Per-kernel commits make `run-all`'s revert clean and the history bisectable. The orchestrator never
sees a diff or a pytest log — only the per-tier conclusions.

### Phase 5 — Report

`migration/report.md`:
- **Summary:** per tier — migrated / failed / skipped-deferred + totals.
- **Per kernel:** status, tests run + pass/fail/hang, **diff size** (call-site lines removed = the
  helper's payoff), perf delta vs the bake-off baseline if measured, refactor notes.
- **Coverage gaps:** kernels migrated with low/no verified coverage — flagged "migrated but
  under-tested" (the riskiest items).
- **Deferred:** what wasn't migrated + why.
- **Provisional confirmations:** results of the in-context checks from Phase 1.

### Checkpoint — after Phase 5
Hand over `migration/report.md`. Summarize the totals, the headline failures (if any), and the
coverage-gap risk list. In `halt` mode, point at the left-in-tree diff + triage for the kernel that
stopped the run. Wait for the user's read.

---

## Output directory contract

```
migration/
├── test_map.md / test_map.json     (Phase 2 — device-verified kernel→test map; KEPT + reused)
├── tiers.md                        (Phase 3 — ordered worklist)
├── report.md                       (Phase 5 — running + final report)
└── log/<kernel>.md                 (Phase 4 — per-kernel diff + tests + result, written by subagents)
ttnn/cpp/ttnn/kernel_lib/<helper>.hpp   (Phase 1 — the materialized helper)
tests/.../test_<helper>.py              (Phase 1 — helper unit test, ported from the bake-off harness)
```

---

## Anti-patterns to avoid
```
❌ Migrating before the kernel→test map exists. Flying blind on regressions.
❌ Trusting a static grep that "a matmul test covers this kernel." REQUIRE device verification
   (build cache / marker) — many op tests never hit the specialized (e.g. sharded mcast) path.
❌ One giant commit for the whole rollout. Per-kernel commits make revert + bisect possible and are
   what run-all mode needs to keep the tree green.
❌ Migrating a kernel with no verified coverage and calling it done. Flag it as a risk.
❌ Migrating deferred kernels "to be thorough." They were scoped out for real reasons.
❌ In halt mode, silently reverting the failed diff. It STAYS so it can be debugged.
❌ Inventing a default --mode. It is always supplied by the caller (ask at Gate 0 if absent).
❌ Migrating inline in the orchestrator. Per-kernel diffs + build/pytest/triage output would blow up
   its context across dozens of kernels. Phase 4 is one subagent PER TIER; only compact conclusions
   return — the verbose record lives in migration/log/ + report.md on disk.
❌ Running tiers (or kernels) in parallel. They share the device + JIT build — strictly sequential,
   even though each tier gets its own subagent context. Subagents isolate context, not execution.
```
