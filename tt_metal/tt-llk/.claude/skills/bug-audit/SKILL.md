---
name: bug-audit
description: Large-scale, resumable bug audit of any repository — every in-scope file read in full by a hunter agent, every candidate adversarially verified, coverage proven rather than claimed, and recall measured against the repo's own past bugs. Hunts the universal bug classes plus domain classes, prioritised by knowledge mined from the repo's real bug history (closed issues, fix PRs and whether those fixes held, review threads). Ships mined knowledge for tt-metal and tt-llk, and the mining pipeline to build it for any other repo. Use for a full or area-wide bug hunt, a diff audit of what landed since a commit, or to (re)build a repo's knowledge pack.
user_invocable: true
---

# /bug-audit — exhaustive, knowledge-weighted bug hunt

## What this is
A hunt for real, reachable defects: wrong results, crashes, hangs, races, leaks, broken contracts. It is not a
style or perf review. Synchronization hazards in Tenstorrent LLK/dataflow code get a deeper, HW-grounded pass from
`race-audit-all` and its sub-audits. This skill covers breadth: all bug classes, across a whole tree. It hands
suspicious sync sites to those audits instead of settling them shallowly.

Everything lives in two places:
- **this skill directory**: the method, the engine (`engine/`), the mining pipeline (`mining/`), the class lists
  (`references/`) and the mined repo packs (`packs/`);
- **a run directory** (you choose it; use shared storage, not `/tmp`): the state of one audit. It holds batches,
  findings, verdicts, dispositions and reports. The run directory is never committed.

## The knowledge stack (read before hunting, and hand it to every hunter)
1. `references/classes-universal.md`: the floor. Always hunted, in every repo.
2. Domain classes, when the repo matches. For Tenstorrent code, use `references/classes-tenstorrent.md`.
3. The repo pack, if one exists: `packs/<repo>.md`. It holds class weights from the repo's real bug history, hot
   spots per area, "fix seeds" (past defects whose pattern may recur elsewhere), fixes found incomplete, and the
   checks reviewers apply.
4. An optional private overlay: a `packs-private/<repo>.md` next to the run directory, for internal-only material
   that must never be committed to a public repo.

**History re-weights; it never removes.** The pack decides what to check first and where to spend effort. A
universal class with no recorded history is still hunted, because history only contains the bugs somebody noticed.
Knowledge is a floor, not a ceiling: hunters report any grounded defect, whether or not a class names it.

No pack for this repo? Build one first (see *Mining a repo*), or run with the class lists alone and say so.

## Modes
| Mode | Scope | How |
|---|---|---|
| full | every source file in the repo | `init_run.py` with priority globs, then the wave loop |
| area | a subtree | `init_run.py --prio 'A=<subtree>/**' --exclude ...` |
| diff | files changed since a commit | `init_run.py --since <commit>`, the cheap way to keep an old full audit current |
| bench | real past bugs, at the commit before their fix | `engine/bench.py prepare/score`, which measures recall |
| mine | build or refresh a repo pack | `mining/` pipeline (below) |

Full, area and bench runs fan out through the **Workflow** tool. That needs the user's explicit opt-in to
multi-agent orchestration. Get it, and state the cost first. Calibration from earlier runs: about 50k tokens per
file audited, and about 70M tokens per 120-batch wave. Verification is most of the agent count.

## Start of every audit: ask the user
Before `init_run.py`, ask these in one AskUserQuestion call. Record the answers in the run's state (the execution tier
through `exec_tier.py configure`), and never assume a default for the execution tier.
1. **Scope:** full repo, an area (globs), or a diff since a commit.
2. **Execution tier (OPTIONAL, off unless the user says yes).** Should the audit also build, analyse and test, to
   catch what reading cannot? Options:
   - **No:** static only (the default answer; nothing is built or run).
   - **Build flavours:** compile every configuration the user names, and turn compiler and linker diagnostics into
     leads. A non-default flavour exposes bugs the default build hides.
   - **Static analyzers and sanitizers:** clang-tidy with the repo's own config, and ASan/UBSan/TSan builds.
   - **Existing tests:** runs the tests that reference each batch's files. This needs the target hardware, and the
     user must confirm which machine and card can be used.
   If yes, ask for the exact commands (or confirm the presets below), the machine, and a time budget. The tier runs
   the repo's code with those commands, so confirm it is acceptable on this machine.
3. **Budget:** tokens and wall-clock. State the calibration (about 50k tokens per file hunted, plus about 40% for
   verification), and whether to run the optional second pass.

tt-metal presets for the execution tier (confirm with the user; they take a clean build dir and many minutes each):
```
exec_tier.py --run <run> configure \
  --build 'release=./build_metal.sh --build-tests' \
  --build 'asan=./build_metal.sh -b ASan --build-tests' \
  --analyze 'clang-tidy=run-clang-tidy -p build_Release -quiet $(git ls-files "*.cpp" | grep -E "^(tt_metal|ttnn)/")' \
  --test-cmd 'pytest -x -q {tests}' --test-root tests/ttnn --test-root tests/tt_metal --max-tests 6 --timeout 7200 \
  --reset-cmd 'tt-smi -r 0'   # a hung test wedges the board; reset between test groups
exec_tier.py --run <run> run        # before the first wave; re-run after re-pointing the tree
```
The signals land in `<run>/exec/signals/<batch>.json`, and `next_wave.py` hands them to the hunters automatically.
A signal is a lead, never a finding: the hunter triages it and the normal verification applies.

## Running an audit
Engine scripts take `--run DIR` (or `BUG_AUDIT_RUN`); paths below are relative to this skill directory.

1. **Pin the tree.** Make a dedicated worktree at the commit to audit
   (`git worktree add --detach <path> <commit>`), and keep it until the run is closed, so every recorded
   `file:line` stays valid.
2. **Init:** `engine/init_run.py --root <tree> --out <run> --repo owner/name --prio 'A=<highest-value globs>' ...
   --knowledge references/classes-universal.md,<domain>,packs/<repo>.md`. Put device kernels and core runtime
   first, host periphery next, and tests and models last. Batches are at most 20 files and 3,500 lines (1,500 for
   priority A), so a hunter can read every line.
   - **Submodules:** `git ls-files` lists a submodule as ONE entry, so by default its files are silently out of
     scope. `init_run.py` warns and names them. Either pass `--recurse-submodules`, or audit each submodule as its
     own run and say so in the report. An earlier whole-repo audit missed an entire submodule this way.
   - **Prior runs:** pass `--prior-run <old run dir>` (repeatable). Each batch then gets what earlier runs
     confirmed or refuted in its files, so refuted findings are not re-raised without new evidence, and
     confirmed ones are not re-reported.
3. **Wave loop:**
   - `engine/next_wave.py --run <run> 45 A > args.json`. This prints the Workflow args and marks the batches in
     flight.
   - Run `Workflow` with the script `engine/audit-wave.js` (pass its content as `script` on first use; reuse the
     returned `scriptPath` after that) and `args` = that JSON.
   - **The moment it finishes:** copy the task output into `<run>/raw_wave_outputs/`, then run
     `engine/persist_wave.py --run <run> <copy>`. Skip this and the wave's verdicts are lost.
   - `engine/consolidate.py --run <run>`, then `engine/status.py --run <run>`. Snapshot the run directory.
   **For anything longer than a wave or two, run the whole loop unattended instead:**
   `tmux new -d -s audit "python3 engine/run_headless.py --run <run> --wave-size 45 --prios A"`. It runs every
   wave in headless sessions that resume on failure, then persists and consolidates each one, and stops when
   nothing is pending. Re-run the same command to resume after a crash. It never loses a finished agent.
4. **Repeat until `status.py` shows 0 pending and 0 in flight.** Batches the read check sent back are re-issued
   first, automatically.
5. **Close the verification gaps:** run `engine/recheck.py --run <run> queue`, then `engine/recheck-wave.js`, then
   `recheck.py persist`, then `recheck.py report`. This rechecks every uncertain or needs-recheck candidate, and a
   10% seeded sample of the refuted ones. If the sample's reversal rate is material (more than about 1 in 10),
   recheck the whole refuted pile.
6. **Dedup, so each bug is filed exactly once.** Two separate steps:
   - **Within the run:** `engine/dedup.py --run <run> inputs`, then `engine/dedup-wave.js`, then
     `dedup.py persist <output>`, then `consolidate.py`. The same defect reported at several lines becomes one entry.
     The same defect in any number of architecture or platform copies (Grayskull, Wormhole, Blackhole, Quasar,
     or a repo's own variants, added with `--variant`) is MERGED into ONE entry, never dropped:
     - the entry lists every site, with each copy's own failure mode and suggested fix, since copies can differ;
     - its severity is the worst across its sites;
     - tables show every site (`file:line`, plus `+ file:line` for each copy);
     - it is filed as one issue naming all the sites, and `OPEN.md` counts it once.
     The merged-in rows stay in `CONFIRMED.md`, marked "merged into".
   - **Against GitHub:** `engine/filed_check.py --run <run> candidates`. It searches the repo's issues and PRs, all
     authors, open and closed, by file name and by the finding's identifiers. Then run `engine/filed-wave.js`, where a
     judge decides whether any candidate is the same defect, then `filed_check.py persist <output>`, then
     `consolidate.py`. It records one of:
     - `already_filed`: an open issue or PR covers it, so it is never filed again;
     - `filed_closed`: reported and closed, but still present in the audited tree (an incomplete fix or a
       regression), so it goes under "needs attention" with the link;
     - `fixed_upstream`: a PR merged after the audited commit fixes it.
     A search failure stops the run; it never silently means "not filed".
7. **Optional second pass over the highest-priority areas.** Independent hunts miss different bugs, but the measured
   gain is small: three post-fix passes together found 49%, against 45% for the best single pass. It is worth it only
   for the areas that matter most: priority-A batches (device kernels, core runtime, the pack's hot areas).
   Re-issue each batch once more to a fresh hunter with `next_wave.py --run <run> 45 A --second-pass`, then run
   the usual workflow and persist. persist_wave.py archives the first hunt, merges the verdicts, and consolidate
   dedupes them by `file:line`.
8. **Measure recall** on the repo's held-out bugs (`packs/<repo>-holdout.jsonl`). Run
   `bench.py prepare --knowledge ...` and a second prepare without knowledge, run a wave loop on each, then
   `bench.py score`. Report both numbers. A run without a recall number has no measured miss rate. Say so.

### Contracts that make the result trustworthy
- **The contract trace is recorded and audited.** Each hunter returns a ledger of every boundary it traced (its
  site, the other side's `file:line`, the verdict, and what it compared), plus the boundaries it skipped and why.
  `persist_wave.py` checks that every "other side" is a real `file:line`. Before screening, an independent trace
  auditor re-traces a sample of each ledger: the first, middle and last "consistent" entries, plus every mismatch
  that has no finding. Whatever it finds joins the candidates. The persist summary reports how many verdicts the
  audit overturned. A high rate means the hunters are skimming.
- **Less code per hunter where it matters.** Priority-A batches default to 1,500 lines (`--batch-lines A=1500`), the
  rest to 3,500. The post-fix miss analysis found that hunters read 27 of 29 missed bugs but only skimmed them.
- **Coverage is proven, not claimed.** Each hunter reports every file's line count and last non-blank line.
  `persist_wave.py` checks both against the tree, and a mismatch sends the batch back. `ledger.tsv` has one row per
  in-scope file (pending, reread or audited, plus its confirmed-finding count) and is updated on every persist. The run is not done while
  any batch is pending or in flight. A partial run is reported as "N of M files", never as "exhaustive".
- **Three-valued verification.** Verifiers answer confirmed, refuted (with the line that makes it safe), or
  uncertain. A dead verifier is unknown, never a refutation. Uncertain findings are surfaced in `UNCERTAIN.md`:
  never filed, never dropped.
- **Nothing is annotated by hand.** `CONFIRMED.md`, `OPEN.md`, `UNCERTAIN.md` and `REFUTED.md` are regenerated.
  Outcomes go through `engine/disposition.py set <file:line> <state> [--pr N --issue N] -m ...`. Run
  `disposition.py sync` at the start of any fix session, and `disposition.py check` after re-pointing the tree.
- **Staleness.** Findings cite the pinned commit. Before filing or fixing, re-check the line on current main.
  To extend an old run, do a diff-mode run from its commit rather than re-auditing everything.

## Mining a repo (building `packs/<repo>.md`)
The same pipeline built the shipped packs. It is repo-agnostic.
1. **Fetch:** `mining/fetch_repo.py owner/name <mine>/raw` (all closed issues and PRs; resumable; weekly windows).
2. **Build cases:** `mining/build_cases.py --issues '<mine>/raw/issue/*.jsonl' --prs '<mine>/raw/pr/*.jsonl'
   --git <clone> --out <mine>/cases.jsonl` [`--xref-git <clone>` when fixes landed in another repo]. This links each
   bug issue to its fix commits (closing PR, `closes` references, commits citing the issue) and to later history:
   reverts, re-fixes, later fix-like commits to the same files.
3. **Triage everything cheaply:** run `mining/make_chunks.py`, then the `mining/triage-wave.js` workflow, then
   `mining/persist_mining.py`. Every case gets verdict, class, component, symptom and deep-read priority.
4. **Pick a holdout set first.** `mining/select.py holdout --git <clone> [--exclude <older holdouts>] [--deep <deep
   stores>]` oversamples (about 1.4x the target) a
   seeded random set of confirmed code bugs with small, code-only fixes. `mining/holdout-screen-wave.js` then asks of
   each one whether the pre-fix code was really defective, and `select.py screened` keeps the first N valid cases in
   `packs/<repo>-holdout.jsonl`. Exclude that set from everything after this step. Skipping the screen let 9 of 75
   non-defects (cleanups, lint fixes, feature enablement) into the first benchmark. A case later found invalid is
   marked `"valid": false` (never deleted), and `bench.py` skips it. Exclusion works by fix COMMIT, not only by case
   id, because an issue and its PR are separate cases that share one fix. A pick sharing a fix with an older holdout
   or a deep-read case is contaminated; mark it `"exclude": true` if one slips through. A pack that has seen the benchmark
   answers scores a meaningless 100%.
5. **Deep-read** the priority cases: run `mining/select.py deep --exclude <holdout>`, then `mining/deep-wave.js`. For each case:
   - the root cause;
   - whether the fix was complete (judged from later history and the current tree, never from the fact that it
     merged);
   - unfixed siblings in the current tree;
   - the general audit check that would have caught it.
6. **Mine reviews:** run `mining/fetch_reviews.py` on PRs with review threads, then `mining/make_review_chunks.py` (a
   loose keyword filter that drops nits before any agent sees them), then `mining/review-wave.js`. This
   gives the defects reviewers caught before merge, and the lessons in PRs closed without merging.
7. **Synthesise:** `mining/synthesize.py` computes the class weights and hot spots. Then write the pack by hand:
   a weighted class table, hot spots, seeds, incomplete fixes, and reviewer checks. Unfixed siblings from step 5
   are candidate bugs. Queue them into an audit run's `recheck.py`-style verification; never file them straight
   from mining.

Every text in a committed pack must be public-safe: no internal hardware internals, no private-doc content, no
issue or PR numbers. Mechanisms are stated in general terms and cite living docs or code. Put anything else in the
private overlay.

## Filing what the audit finds
- **Repro before a fix PR.** A static finding becomes a PR only after its failure has been reproduced against
  shipped code, or when the fix is verifiable without the target hardware (compile errors, host logic with a unit
  test). Otherwise file it as an issue.
- **File only what `OPEN.md` still lists after step 6.** Those findings have no match in any issue or PR, by any
  author. One arch-copy entry is ONE issue, and it names every site (the Wormhole and the Blackhole `file:line`), so
  neither copy is lost and neither is filed twice. Record the disposition (`disposition.py set ... --issue N`) the
  moment an issue is filed: a finding with no disposition is assumed unfiled, and a second session would file it
  again.
- **Public repos get no hardware internals:** no RTL signal names, `.sv` paths or waveform decodes in issues, PRs or
  comments. Internal detail goes to the internal tracker. Ask before anything is pushed or posted.
- Issue bodies:
  - Put the false-positive disclaimer right after the header.
  - Carry technical content only: location and permalink, failure scenario, evidence, suggested fix, and a
    `- [ ] fixed` checkbox.
  - Never describe the audit method, the vote counts or the batch ids.
  - Never discuss ownership or name people. A `CODEOWNERS:` routing line is fine.
- File HIGH severity only unless the user asks otherwise. File unassigned. Titles are the user's to edit; never
  bulk-edit filed titles. Keep each body under GitHub's 65,536 characters; if an area has more, cluster by defect
  mechanism, one issue per cluster. Mention team handles in backticks, not with @, so filing doesn't email a team.
- **A fix PR** comes with a regression test that fails without the fix and passes with it, and it counts as done only
  when CI passes on EVERY gate platform, not just locally.
- **Ask before anything public:** filing, commenting, pushing.

## Measured recall (2026-09-26, held-out real bugs audited at the commit before their fix)
Scored SEMANTICALLY (`bench.py judge-inputs`, then `judge-wave.js`, then `bench.py judged`). Two blind judges per case
agreed on 809 of 810 findings. The earlier line-proximity scores overstated some arms by up to 11 points and are not
used. Nine of the 75 held-out "bugs" were not defects in the pre-fix code and are excluded.

| Repo (valid held-out bugs) | Universal classes | + Tenstorrent classes | + mined pack (full skill) |
|---|---|---|---|
| tt-metal (53), first run | 28% | 36% | 32% |
| tt-metal (53), after the post-mortem fixes | **42%** | **45%** | 38% |
| tt-llk (13), first run | 38% | 38% | 38% (54% counting unconfirmed candidates) |

What the data supports:
- **The post-mortem fixes are the one significant, measured gain.** The contract-trace hunter step and the new
  classes took the universal-only arm from 15 to 22 of 53 (8 gained, 1 lost; exact McNemar p ≈ 0.04).
- **The mined pack shows no measurable recall benefit, in either run.** Before the fixes it scored 5 vs 3 against
  the baseline (p ≈ 0.7). After them it scored 1 vs 5 against the domain arm (p ≈ 0.2): not significant, but not
  better. The "significant pack win" first reported was a line-proximity artifact. The pack arm does confirm the
  most findings in the same files (145, against 128 and 116), but whether those extra findings are real is
  unmeasured.
- **The mining still paid off indirectly.** The classes and the contract-trace rule behind the measured gain came from
  a post-mortem of what the audit missed on the mined held-out bugs. As a benchmark and a source of lessons, the
  history is valuable. As text for hunters to read up front, it has not shown value.
- **Extra passes help less than line proximity suggested.** The three post-fix arms together found 26 of 53 (49%),
  against 45% for the best single arm. All six runs together found 30 (57%).
- **Noise is about ±5 cases per run,** so differences smaller than that between single runs mean nothing.
- A transcript diagnostic of the discordant cases found no case where the pack distracted or misled a hunter.
  Hunters spent only about 5-20% of their tool calls on it.
- **One defect per finding:** a finding that bundles a real defect with a wrong claim gets refuted whole.

**Why the post-fix audit still misses about 55%** (post-mortem of the best arm's 29 misses):
- **The hunter read the buggy lines in 27 of 29 cases, but only skimmed them.** It did the required contract trace
  fully in just 2 cases (partly in 20, not at all in 5). The misses are about depth and diligence per site, not coverage.
- **For 11 of 29, execution is the cheapest reliable catch, not reading:** a compile of a non-default build flavour
  (5), a sanitizer or static analyzer (4), or an existing test on the target arch (2). A static-only audit has a
  ceiling that more knowledge will not lift.
- For 16, the analysts proposed a narrow prompt rule or class. Adding those one by one would overfit: the rules
  come from the benchmark's own cases, so any "gain" would be measured on the answers. Needed instead:
  - **structural enforcement:** the hunter's output lists each boundary with the other-side `file:line` it checked,
    so a skipped trace is visible and verifiable;
  - **less code per hunter:** smaller batches, or a focused per-function pass on the highest-priority files;
  - **a fresh holdout** for measuring any rule derived from these misses.
- Only 1 of 29 is genuinely not findable from source or execution.

**The shipped version, measured on a FRESH holdout** (`packs/tt-metal-holdout-v2.jsonl`: 50 screened bugs, none in the
first holdout or among the deep-read fixes, 3 excluded as contaminated). Static only, full skill (universal +
Tenstorrent classes + pack, the contract-trace ledger with its trace audit, and the post-mortem hunter rules):
**23 of 47 = 49%**, scored semantically, with the two judges agreeing on every finding. It is not directly
comparable to the 45%/38% above, which were measured on a different set of bugs.
The run cost about 880 agents for 50 batches (~17.7 per batch; waves are capped accordingly). The trace audit
overturned 24 of 223 re-checked "consistent" verdicts (11%), so hunters' own trace verdicts are wrong about one time
in nine. Verification confirmed 260 of 261 candidates. **Precision, measured separately:** 24 of a random 25 of those
confirmations (the non-benchmark ones) were judged real and reachable by two independent reviewers (96%, 95% CI
about 80-99%), with the reviewers agreeing on all 25. 19 of the 25 were later fixed on main by other commits.
Trace-audit findings scored 9 of 10, own-hunt 10 of 10.

Built from that post-mortem, and NOT yet measured in isolation: the recorded and audited contract-trace ledger, smaller
priority-A batches, the "look hard / one defect per finding" hunter rules, and the optional execution tier. These
were designed from the current holdout's misses, so measure them on a FRESH holdout (`select.py holdout --seed <new>`
plus the screen, excluding the old holdout), not on the 53 cases they were derived from.

**Why the rest were missed** (post-mortem of all 45 misses; details in the benchmark run's `miss-analysis/REPORT.md`):
- **Cross-file tracing not done (13 cases), and the defect lived outside the batch (6):** the largest group. The
  evidence was one hop away, and the hunter did not take that hop. Examples: a caller's `uint32_t` narrowed into a callee's
  `uint8_t` parameter that is packed into a 4-bit field; `validate()` accepting layouts the program factory does not
  implement; a renamed keyword argument that callers still pass; unpack and math MOPs disagreeing on dvalid counts.
- **Class missing from the lists (6), or domain knowledge missing (4):** examples are symbol visibility across
  shared libraries, preprocessor-macro collisions and `#elif` on undefined macros, namespace lookup shadowing,
  `.begin()` on a possibly-empty container, stall-resource versus stall-condition constants, and register-literal
  consistency.
- **Hunt variance (5):** another arm found the bug and this one did not. A second pass recovers these.
- **Attention (3):** the site was read but a low-salience defect went unnoticed (a dead local, a wrong diagnostic).
- **Verifier kills (2, both tt-llk):** each finding was right but was held back because the library entry point had no
  in-tree caller. The callers live in another repo.
- **Not statically visible (2):** a compiled stack-frame overflow, and an undocumented NoC erratum.

Applied after the post-mortem, and re-measured (see the table above):
- a mandatory one-hop CONTRACT TRACE, sibling comparison and mechanical sweeps in the hunter prompt;
- the missing classes (`sibling-divergence`, `dead-store`, `empty-access`, `lossy-compare`, `validate-vs-impl`,
  `symbol-resolution`, `preprocessor`, `linkage-visibility`, and the `tt-` classes for dvalid balance, stall
  operands, register literals, hardcoded arch constants, core-flavour maps, JIT defines and dispatch field width);
- a verifier rule that library entry points are reachable by default;
- a stricter triage definition of "code bug", a holdout validity screen, and deep-read verdicts overriding triage in
  the weights;
- the second-pass step in the run loop.
To re-measure after any change to the prompts, classes or packs: `bench.py prepare` into fresh run directories,
then `score --cases <holdout>`. Do NOT edit any knowledge file while a run is in flight: hunters read them at start
time, so an edit mid-run measures a mixture. For long runs, launch each arm with `engine/run_headless.py
--run <arm> --bench-cases <holdout>` in tmux (a plain `claude -p` exits after about 10 minutes and kills its workflow).

## Lessons from earlier large audits
- **Hunt, don't fill a checklist.** A candidate-list-driven pass found nothing in a tree where a method-driven hunt
  found dozens. Recall tools and packs augment the hunt; they never replace reading the code.
- **Coverage is `files x classes`.** A run that reads every file but drops a class is bounded. Record the knowledge
  files each run used, and give runs made with an older pack a diff hunt for the classes added since.
- **"Confirmed" by agreement is not a repro.** Say "statically confirmed". Reachability is the most common reason a
  plausible finding dies. Check host-side validation, the program factory and `constexpr` before calling a
  value-gated path reachable.
- **Incomplete fixes cluster in siblings.** One arch, dtype or overload gets fixed and its copies do not. But an
  identical line on another variant is not automatically the same bug: compare each variant with its own
  counterpart.
- **Batch sizing:** tiny batches waste most of the budget in per-agent overhead, so pack directories contiguously
  up to the line bound.
- **The 1,000-agent cap per workflow.** Measured on the final version: about 17.7 agents per batch (hunt, trace
  audit, screens including trace-audit candidates, two deep verifiers per survivor). So a wave must stay at 50
  batches or fewer (`audit-wave.js` refuses larger ones). If verification is per finding, cluster it by (class, file) when a
  wave would exceed the cap.
- **Yield steers the run.** Order later waves by measured confirm yield per class (`COVERAGE.md`), not by class
  number. A class with many candidates and a near-zero confirm rate is a PROMPT defect: fix the prompt, not the code.
- **"No caller" is not "unreachable".** A parser saying a file or function is dead is not proof (earlier audits found
  instantiations hidden inside comments and regex misses). Deprioritise apparently dead code; never exclude it
  silently.
- **Never persist verdicts by hand.** Always go through `persist_wave.py`. A hand-persisted wave once contaminated
  two unrelated in-flight batches.
- **Hunters do not execute.** Workflow agents have run on-device experiments without being asked. Static hunts are
  told not to build, run or touch hardware; execution happens only in the opt-in tier.
- **Workflow mechanics:** a thrown workflow returns `[]`, but its journal survives on disk, so resume with
  `resumeFromRunId` (`run_headless.py` does this). Read results from the task's output file, never from the
  completion notification, which truncates large returns. The Workflow tool refuses some script paths: pass the
  script inline once, then reuse the `scriptPath` it returns. Never launch with placeholder args.
- **A missing document is never evidence** of a negative or a positive: "the docs don't say X is ordered" proves
  nothing either way. Hardware-behaviour verdicts cite the authoritative source, or they are "uncertain". Staged verification (one screener, then two deeper lenses for survivors only) cuts the
  verify tail, which was 80% of agents.
