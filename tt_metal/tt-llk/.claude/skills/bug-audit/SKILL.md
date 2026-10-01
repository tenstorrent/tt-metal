---
name: bug-audit
description: Large-scale, resumable bug audit of any repository — every in-scope file read in full by a hunter agent, every candidate adversarially verified, coverage proven rather than claimed, and recall measured against the repo's own past bugs. Hunts the universal bug classes plus domain classes, and sweeps the unfixed sibling copies of past fixes mined from the repo's real bug history (closed issues, fix PRs and whether those fixes held, review threads). Ships mined knowledge for tt-metal and tt-llk, and the mining pipeline to build it for any other repo. Use for a full or area-wide bug hunt, a diff audit of what landed since a commit, or to (re)build a repo's knowledge pack.
user_invocable: true
---

# /bug-audit — exhaustive, knowledge-weighted bug hunt

## What this is
A hunt for real, reachable defects: wrong results, crashes, hangs, races, leaks, broken contracts. It is not a
style or perf review. Synchronization hazards in Tenstorrent LLK/dataflow code get a deeper, HW-grounded pass from
`race-audit-all` and its sub-audits. This skill covers breadth: all bug classes, across a whole tree. It hands
suspicious sync sites to those audits instead of settling them shallowly.

Engine tests: `python3 -m pytest tests -q --noconftest` (hermetic: synthetic run directories, no network, no agents).
Run them after changing the engine or the mining scripts.

Everything lives in two places:
- **this skill directory**: the method, the engine (`engine/`), the mining pipeline (`mining/`), the class lists
  (`references/`) and the mined repo packs (`packs/`);
- **a run directory** (you choose it; use shared storage, not `/tmp`): the state of one audit. It holds batches,
  findings, verdicts, dispositions and reports. The run directory is never committed.

## The knowledge stack
Handed to every hunter:
1. `references/classes-universal.md`: the floor. Always hunted, in every repo.
2. Domain classes, when the repo matches. For Tenstorrent code, use `references/classes-tenstorrent.md`.

Not handed to hunters by default:
3. The repo pack, if one exists: `packs/<repo>.md`. It holds class weights from the repo's real bug history, hot
   spots per area, "fix seeds", fixes found incomplete, and the checks reviewers apply. As reading for hunters it
   showed **no measurable recall benefit** in either benchmark run (`references/measurement-history.md`), so it
   costs tokens on every batch for no measured gain. Add it only to re-measure it, with
   `--knowledge references/classes-universal.md,<domain>,packs/<repo>.md`: an explicit list replaces the default,
   so name the class lists too. Its hot areas still set the batch order: `init_run.py` finds the pack for `--repo`
   and moves files in those directories, if no `--prio` glob claimed them, to priority A (`--pack none` turns
   this off).
4. An optional private overlay: a `packs-private/<repo>.md` next to the run directory, for internal-only material
   that must never be committed to a public repo.

**The history pays off in three other ways, and those are measured.** The held-out bugs are the recall benchmark.
The post-mortem of what the audit missed on them produced the classes and the contract-trace rule behind the one
significant gain. And the deep reads of past fixes name their **unfixed siblings** in the current tree, which the
sibling sweep verifies (*Sweeping siblings from history*): on tt-metal that confirmed 536 of 776 leads. A universal class with no recorded history is still hunted, because history only contains the bugs
somebody noticed. Knowledge is a floor, not a ceiling: hunters report any grounded defect, whether or not a class
names it.

No mined history for this repo? Run with the class lists alone and say so; mine it (see *Mining a repo*) to get a
benchmark and a sibling sweep.

## Modes
| Mode | Scope | How |
|---|---|---|
| full | every source file in the repo | `init_run.py` with priority globs, then the wave loop |
| area | a subtree | `init_run.py --prio 'A=<subtree>/**' --exclude ...` |
| diff | files changed since a commit | `init_run.py --since <commit>`, the cheap way to keep an old full audit current |
| siblings | the unfixed copies of past fixes, from mined deep reads | `engine/siblings.py from-deep`, then the recheck verification (*Sweeping siblings from history*) |
| bench | real past bugs, at the commit before their fix | `engine/bench.py prepare/score`, which measures recall |
| mine | build a repo's mined history and pack, once | `mining/` pipeline (below) |
| refresh | only what closed since the last mining | `mining/marker.py delta`, then `fetch_repo.py --since` (*Refreshing the mined history*) |

Full, area and bench runs fan out through the **Workflow** tool. That needs the user's explicit opt-in to
multi-agent orchestration. Get it, and state the cost first. Measured: one 50-batch wave took about 37M tokens and
881 agents. A full tt-metal run (about 18,000 files and 4.4M lines with the default extensions, so about 1,250
batches) is therefore about 25 waves: roughly 0.9B tokens and 22,000 agents. That wave was 1-3 file benchmark
batches, and a real batch gives each hunter more code to read, so treat these as a floor. Verification is most of the
agent count.

## Start of every audit: ask the user
Before `init_run.py`, ask questions 1-4 in one AskUserQuestion call, then question 5 in a second call: its cost
depends on the scope answer, and one call takes at most four questions. Record the answers in the run's state (the execution tier
through `exec_tier.py configure`), and never assume a default for the execution tier.
1. **Scope:** full repo, an area (globs), or a diff since a commit.
2. **Execution tier (OPTIONAL, off unless the user says yes).** Should the audit also build, analyse and test, to
   catch what reading cannot? Options:
   - **No:** static only (the default answer; nothing is built or run).
   - **Build flavours:** compile every configuration the user names, and turn compiler and linker diagnostics into
     leads. A non-default flavour exposes bugs the default build hides.
   - **Static analyzers and sanitizers:** clang-tidy with the repo's own config, and ASan/UBSan/TSan builds.
   - **Existing tests:** runs the tests that reference each batch's files. This needs the target hardware, and the
     user must confirm which machine and card(s) can be used; the cards go to `exec_tier.py configure --devices`,
     which pins every test to them (`TT_VISIBLE_DEVICES`) and resets only them.
   If yes, ask for the exact commands (or confirm the presets below) and the machine, and state the worst-case time:
   up to the timeout per batch's test group, twice if it fails and is re-run (about 4 hours per batch with the preset
   7,200 s), with no overall cap. The tier runs the repo's code with those commands, so confirm it is acceptable on
   this machine.
3. **Cost, and the second pass.** The audit is exhaustive by design: it runs until every in-scope file is audited,
   with no token or time cap. State the cost for the chosen scope up front (see the calibration above); a user who
   wants to spend less narrows the scope (question 1) rather than capping the run. Ask whether to run the optional
   second pass over priority A.
4. **New history since the last mining.** Run `mining/marker.py delta <mine>/<repo>.mined.json` first (count queries
   only) and report what closed since the watermark. If there is any, offer the refresh (*Refreshing the mined
   history*): it costs agents only for the new cases, and it gives the sibling sweep new leads and the recall
   measurement a fresh holdout.
5. **Sibling sweep** (ask whenever the repo has mined deep reads, `<mine>/<repo>_deep.jsonl`; recommend yes). Should
   the audit also verify the unfixed copies of past fixes that fall inside its scope? This is the part of the mined
   history with a measured payoff (536 of 776 leads confirmed on tt-metal), and it finds bugs the hunt does not.
   Cost: three verifiers per in-scope lead. With the question, give the rough size: count the `unfixed` sibling
   locations under the chosen scope's paths in the deep reads. After init, `--in-scope` prints the exact count.

tt-metal presets for the execution tier (confirm with the user; they take a clean build dir and many minutes each):
```
exec_tier.py --run <run> configure \
  --build 'release=./build_metal.sh --build-tests' \
  --build 'asan=./build_metal.sh -b ASan --build-tests' \
  --analyze 'clang-tidy=run-clang-tidy -p build_Release -quiet $(git ls-files "*.cpp" | grep -E "^(tt_metal|ttnn)/")' \
  --test-cmd 'pytest -x -q {tests}' --test-root tests/ttnn --test-root tests/tt_metal --max-tests 6 --timeout 7200 \
  --devices <confirmed ids> --reset-cmd 'tt-smi -r {devices}'   # a hung test wedges the board; reset between groups
exec_tier.py --run <run> run        # before the first wave; re-run after re-pointing the tree
```
The signals land in `<run>/exec/signals/<batch>.json`, and `next_wave.py` hands them to the hunters automatically.
A signal is a lead, never a finding: the hunter triages it and the normal verification applies.

## Running an audit
Engine scripts take `--run DIR` (or `BUG_AUDIT_RUN`); paths below are relative to this skill directory.

1. **Pin the tree.** Make a dedicated worktree at the commit to audit
   (`git worktree add --detach <path> <commit>`), and keep it until the run is closed, so every recorded
   `file:line` stays valid.
2. **Init:** `engine/init_run.py --root <tree> --out <run> --repo owner/name --prio 'A=<highest-value globs>' ...`.
   Put device kernels and core runtime first, host periphery next, and tests and models last. Batches are at most
   20 files and 300 lines (a longer file is a batch of its own), so a hunter can read and trace every line.
   `--knowledge` defaults to the universal classes, plus `classes-tenstorrent.md` for a `tenstorrent/` repo; pass
   it to name another domain list, or `none` to run without a class list.
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
   **Sibling sweep, if the user said yes:** `engine/siblings.py --run <run> from-deep <mine>/<repo>_deep.jsonl
   --in-scope`. The in-scope leads enter the run as uncertain `history-sibling` findings, so step 5 verifies them with
   the rest. Their severity is a placeholder: re-rate the confirmed ones with `engine/severity-wave.js` in step 6.
5. **Close the verification gaps:** run `engine/recheck.py --run <run> queue`, then `engine/recheck-wave.js`, then
   `recheck.py persist`, then `recheck.py report`. This rechecks every uncertain or needs-recheck candidate, and a
   10% seeded sample of the refuted ones. If the sample's reversal rate is material (more than about 1 in 10),
   recheck the whole refuted pile.
6. **Dedup, so each bug is filed exactly once.** Two separate steps:
   - **Within the run:** `engine/dedup.py --run <run> inputs`, then `engine/dedup-wave.js`, then
     `dedup.py persist <output>`, then `consolidate.py`. The same defect reported at several lines becomes one entry. Groups are directories, plus cross-directory sets of findings that name at least two of the same identifiers (a defect reported at a call site and at its definition), so the judge compares those too.
     The same defect in any number of architecture or platform copies (Grayskull, Wormhole, Blackhole, Quasar,
     or a repo's own variants, added with `--variant`) is MERGED into ONE entry, never dropped:
     - the entry lists every site, with each copy's own failure mode and suggested fix, since copies can differ;
     - its severity is the worst across its sites;
     - tables show every site (`file:line`, plus `+ file:line` for each copy);
     - it is filed as one issue naming all the sites, and `OPEN.md` counts it once.
     The merged-in rows stay in `CONFIRMED.md`, marked "merged into". `file:line` is a finding's identity, so a
     second finding on the same line (the same defect under another class, or a second defect) is kept on that
     entry as "also reported at this line" with its own status, never dropped; split it out if it is a second bug.
   - **Against GitHub:** `engine/filed_check.py --run <run> candidates`. It searches the repo's issues and PRs, all
     authors, open and closed, by file name, by the function enclosing the site, and by the finding's identifiers.
     For a large run pass `--local` with current dumps of EVERY issue and PR, open and closed, of every repo where the
     code lives or lived (`owner/name=<glob>` for another repo): a closed-only dump silently misses the open reports,
     which are exactly the duplicates. Candidates are ranked per repo, so a large repo never crowds out a small one.
     Then run `engine/filed-wave.js`, where a
     judge decides whether any candidate is the same defect, then `filed_check.py persist <output>`, then
     `consolidate.py`. It records one of:
     - `already_filed`: an open issue or PR covers it, so it is never filed again;
     - `filed_closed`: reported and closed, but still present in the audited tree (an incomplete fix or a
       regression), so it goes under "needs attention" with the link;
     - `fixed_upstream`: a PR merged after the audited commit fixes it.
     A search failure stops the run; it never silently means "not filed". This search matches words, so it misses a
     report that describes the behaviour rather than naming the file or function ("fast exp ignores ITERATIONS").
     **Before filing anything, also search for it by behaviour** (the op and the symptom), by hand or with GitHub's
     search; that is how the last duplicates of a large run were found.
   - **Re-rate severity with one rubric:** a hunter picks high / medium / low on its own judgement, and verification
     checks whether a bug is real, not how bad it is, so unrated severities are not comparable across findings.
     `engine/severity.py --run <run> prepare --to-dir <dir>`, then `engine/severity-wave.js` with the args it prints,
     then `severity.py persist <output>`, then `consolidate.py`. Each rating is a disposition severity override with a
     one-line reason (the hunter's rating stays visible as `severity_audit`). File from the rated severities.
   - **Write the suggested fixes:** `engine/fixes.py --run <run> prepare --to-dir <dir>`, then `engine/fix-wave.js`
     with the args it prints, then `fixes.py persist <output>`, then `consolidate.py`. Each open finding and each
     merged site gets a fix grounded in the current code, from the verifiers' write-ups, with a `Test:` line. A
     history-sibling lead has no fix of its own until this step: its scenario and evidence describe the past bug.
7. **Optional second pass over the highest-priority areas.** Independent hunts miss different bugs, but the measured
   benchmark gain was a few points (`references/measurement-history.md`). It is worth it only for the areas that
   matter most: priority-A batches (device kernels, core runtime, the pack's hot areas).
   Re-issue each batch once more to a fresh hunter with `next_wave.py --run <run> 45 A --second-pass`, then run
   the usual workflow and persist. persist_wave.py archives the first hunt, merges the verdicts, and consolidate
   dedupes them by `file:line`.
8. **Measure recall** on the repo's held-out bugs (`packs/<repo>-holdout.jsonl`). Run
   `bench.py prepare --knowledge ...` and a second prepare without knowledge, run a wave loop on each, then
   `bench.py score`. Report both numbers. A run without a recall number has no measured miss rate. Say so.

### Contracts that make the result trustworthy
- **The contract trace is recorded and audited.** Each hunter returns a ledger of every boundary it traced (its
  site, the other side's `file:line`, the verdict, and what it compared), plus the boundaries it skipped and why.
  `persist_wave.py` checks that every "other side" is a real `file:line` (repo-relative in the tree, absolute for a
  file outside it, such as a fetched dependency). Before screening, an independent trace
  auditor re-traces a sample of each ledger: the first, middle and last "consistent" entries, plus every mismatch
  that has no finding. Whatever it finds joins the candidates. The persist summary reports how many verdicts the
  audit overturned. A high rate means the hunters are skimming.
- **Little code per hunter.** Batches default to 300 lines (`--max-lines`; `--batch-lines` sets a budget per
  priority). Depth follows lines per agent: on the same full-size code, hunters found 4 of 4 verified bugs at 300
  lines, 1 at 800 and 0 at the old 1,500-3,500 (*references/measurement-history.md*). It costs about 5x a wave of
  the old batches. The post-fix miss analysis found the same cause: hunters read 27 of 29 missed bugs but skimmed them.
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
2. **Build cases:** `mining/make_cases.py --issues '<mine>/raw/issue/*.jsonl' --prs '<mine>/raw/pr/*.jsonl'
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
   a weighted class table, hot spots, seeds, incomplete fixes, and reviewer checks. The unfixed siblings from step 5
   are candidate bugs: sweep them (*Sweeping siblings from history*); never file them straight from mining.
8. **Mark how far the mining reached:** `mining/marker.py write <mine>/<repo>.mined.json --repo owner/name --dumps
   '<mine>/raw/issue/*.jsonl,<mine>/raw/pr/*.jsonl' --deep <deep.jsonl> --holdout <holdouts> [--tree-commit <sha>]`.
   The watermark is the latest close time in the dumps. A published pack gets its own `packs/<repo>.mined.json`,
   written with `--public` (dates and counts, no issue or PR ids).

## Refreshing the mined history
Mining a repo's whole history is the expensive part, and it is done once. After that, a refresh reads only what
CLOSED since the watermark: nothing mined before is fetched, triaged or deep-read again.
1. **What is new:** `mining/marker.py delta <mine>/<repo>.mined.json` counts the issues and PRs closed since.
2. **Fetch the delta:** `mining/fetch_repo.py owner/name <mine>/raw --since <watermark date>`. It windows on the
   CLOSE date, into `closed_*.jsonl` files beside the full fetch. Windowing on the creation date misses most of it:
   of the 40 tt-metal issues closed in the two days after one watermark, 38 had been opened before it.
3. **Build, triage and deep-read only the new cases:** mining steps 2, 3 and 5 on the `closed_*` dumps, with
   `select.py --exclude` given the existing deep-read store and every holdout (it matches by id AND by fix commit, so
   a case already read under another id is skipped too). Where a new fix touches the files of an old deep-read
   case, re-read that old case too: its fix-completeness verdict may have changed (a revert, a re-fix).
4. **Use the new cases twice.** Their `unfixed` siblings go to the sibling sweep. And bugs fixed after the watermark
   were never seen by the mining, so they are a clean holdout: pick the next recall benchmark from them
   (mining step 4) before they join the deep reads.
5. **Move the watermark:** re-run `marker.py write` over all the dumps once the new cases are persisted. The mined
   store is updated in place, with no approval needed. The committed pack and its `packs/<repo>.mined.json` change
   only when the user agrees, since that is a commit to the repo. The pack text need not be rewritten for a refresh:
   as reading for hunters it has not shown value.
An audit's own findings are not history yet. They become history when they are fixed and closed, and the next
refresh reads them then.

## Sweeping siblings from history
A fix lands in one arch, dtype or overload, and its copies keep the bug. The deep read of each past fix (mining step
5) lists the fix's siblings in the current tree as fixed, unfixed, unsure or not applicable. Every `unfixed` one is a
lead. This finds bugs the hunt does not, because it starts from where history says the defect lives rather than from
whatever file a batch holds, and it needs the deep reads, not the pack.
1. **Pin the tree** at the commit the deep reads judged ("current tree"), as for an audit, and create a run for it
   (`init_run.py`, or reuse an audit run so the leads sit with its findings).
2. **Leads:** `engine/siblings.py --run <run> from-deep <mine>/<repo>_deep.jsonl=<repo> [...]`. One lead per
   `file:line`, carrying every past case that names it; a location given in words keeps a unique key of its own.
   `--include-unsure` adds the `unsure` ones too.
3. **Verify:** `engine/recheck.py --run <run> queue --to-dir <items> --max 330` (three verifiers each; 330 leads is the
   1000-agent cap), then `engine/recheck-wave.js` with the args it prints, then `recheck.py persist`. Repeat until
   nothing is queued. For hundreds of leads run each wave unattended with `engine/run_workflow_headless.py`.
4. **Consolidate, dedup, file-check, re-rate severity and write the fixes** exactly as in step 6 of *Running an
   audit*. The re-rating matters most here: every lead enters with a placeholder medium.
The first sweep, over the tt-metal and tt-llk deep reads (927 unfixed-sibling entries), verified 776 leads: 536 were
confirmed, and 388 of them were new after dedup and the filed check.

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

## What the measurements mean for a run
Recall and precision were measured on the repos' own past bugs; the rounds, their conditions and the post-mortems are
in `references/measurement-history.md`. What matters when running an audit:
- **One pass finds roughly half of the known bugs** (23 of 47 on a fresh holdout, exact 95% interval 34-64%). So zero
  findings in an area does not mean the area is clean. Say so in the report.
- **That is an optimistic figure.** It was measured on benchmark batches of 1-3 files, each known to hold a bug, with
  the pack handed to hunters; the shipped default does not hand out the pack. Real batches are up to 20 files and
  300 lines, and the files are not known to hold a bug, so real recall may differ. Precision (24 of 25 confirmations real)
  was measured on the same small batches; on normal-sized batches it is unmeasured.
- **Most misses are read but not traced:** hunters read the buggy lines in 27 of 29 post-fix misses and skimmed them.
  That is why the contract trace is recorded and audited, and why batches are 300 lines.
- **Independent passes find different bugs.** Run the second pass (step 7) over priority A when those areas matter.
- **Verification earns its cost on real input.** It refuted 28% of candidates in the whole-repo July audit and in the
  sibling sweep, though almost nothing on the tiny benchmark batches.
- **Execution catches what reading cannot:** for 11 of 29 post-fix misses, a build flavour, a sanitizer or a test was
  the cheapest catch (the optional execution tier).

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
- **The 1,000-agent cap per workflow.** About 17.7 agents per batch on average (hunt, trace audit, screens including
  trace-audit candidates, two deep verifiers per survivor), measured on small benchmark batches. Outliers are the
  risk: one real batch once had 144 candidates, about 400 agents on its own. So `audit-wave.js` keeps waves at 50
  batches AND counts every agent: a candidate whose screen and deep verifiers would not fit under 950 is deferred as
  `needs_recheck`, never dropped, and step 5's recheck verifies it.
- **Yield steers the run.** Order later waves by measured confirm yield per class (`COVERAGE.md`), not by class
  number. A class with many candidates and a near-zero confirm rate is a PROMPT defect: fix the prompt, not the code.
- **"No caller" is not "unreachable".** A parser saying a file or function is dead is not proof (earlier audits found
  instantiations hidden inside comments and regex misses). Deprioritise apparently dead code; never exclude it
  silently.
- **Never persist verdicts by hand.** Always go through `persist_wave.py`. A hand-persisted wave once contaminated
  two unrelated in-flight batches.
- **Hunters do not execute.** Workflow agents have run on-device experiments without being asked. Static hunts are
  told not to build, run or touch hardware, and the headless drivers enforce it: their sessions deny builds, test
  runners, card tools and tree-changing commands (`common.STATIC_DENY`), and workflow agents inherit those rules.
  Read-only commands are unaffected. Each wave records how many calls were refused (`blocked_actions` in the run's
  headless state). The rules match command text, so they are a guard, not a sandbox. Execution happens only in the
  opt-in tier, whose commands `exec_tier.py` runs itself.
- **Workflow mechanics:** a thrown workflow returns `[]`, but its journal survives on disk, so resume with
  `resumeFromRunId` (`run_headless.py` does this). Read results from the task's output file, never from the
  completion notification, which truncates large returns. The Workflow tool refuses some script paths: pass the
  script inline once, then reuse the `scriptPath` it returns. Never launch with placeholder args.
- **A missing document is never evidence** of a negative or a positive: "the docs don't say X is ordered" proves
  nothing either way. Hardware-behaviour verdicts cite the authoritative source, or they are "uncertain". Staged verification (one screener, then two deeper lenses for survivors only) cuts the
  verify tail, which was 80% of agents.
