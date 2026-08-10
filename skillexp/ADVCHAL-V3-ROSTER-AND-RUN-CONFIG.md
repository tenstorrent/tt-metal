# advchal-v3 — roster coverage, and the run configuration

Two things `run-record/FUTURE-RUNS.md` says to settle **before any mechanics**, settled here.
Companion to [`ADVCHAL-V3-CHANGES.md`](ADVCHAL-V3-CHANGES.md) and [`ADVCHAL-V3-STEP0.md`](ADVCHAL-V3-STEP0.md).

## 1. Coverage — can the tool under test even see what dominates each cell? (A1)

*"A low advisor contribution on three of four models means the advisor could not see the dominant path, not
that the advice is weak."* Derived from step 0's replay, so it is v2's own data rather than an assumption:
per cell, the layer kind carrying most of model decode time, and how much of that kind the advisor never saw.

| model | arm | dominant kind | % of model | untraced |
|---|---|---|---:|---:|
| north-mini | `fuse-noadvise` | `sliding_attention_moe` | 75.5 % | **69.0 %** |
| north-mini | `nofuse-noadvise` | `sliding_rope_moe` | 74.6 % | **75.7 %** |
| north-mini | `nofuse-noadvise-onA` | `sliding_attention_sparse_moe` | 74.6 % | **76.6 %** |
| gemma-4-26B | `nofuse-noadvise-onA` | `sliding_attention` | 81.9 % | **64.7 %** |
| qwen3.6-27B | `nofuse-noadvise` | `linear_attention` | 97.4 % | **63.5 %** |
| gemma-4-12B | `exp11` | `sliding_attention` | 82.0 % | 4.8 % |
| phi-3.5 | `fuse-noadvise` | `dense` | 100 % | 13.4 % |
| phi-3.5 | `nofuse-noadvise` | `dense` | 100 % | 10.7 % |
| phi-3.5 | `nofuse-noadvise-onA` | `dense` | 100 % | 14.8 % |
| phi-3.5 | `exp17` | `dense` | 100 % | 9.1 % |
| llama-3.1-8B | `exp17` | `dense` | 100 % | 4.5 % |
| qwen3.6-27B | `fuse-noadvise` | `full_attention` | 100 % | 4.5 % ⚠ |

**Five of twelve cells had the advisor blind to roughly two-thirds or more of their dominant kind.** Those
are the coverage zeros, and they are what the v3 tracer handlers exist to close — the corpus measured
75.66 % → 21.12 %, 77.15 % → 14.39 % and 64.70 % → 4.17 % on three of these once the handlers were in.

⚠ **And the last row is the trap this table would otherwise walk into.** qwen `fuse-noadvise` reads as the
cleanest cell in the corpus at 4.5 % untraced — because its `linear_attention` kind, **97 % of that model's
decode time**, produced no reconciliation at all: it was declared tracer-unreachable, so it contributes no
row and the arithmetic silently runs over the 3 % that remains. A cell looks perfect precisely because its
dominant path is missing from the data. That is the same shape as *"a lookup that failed and returned a
reassuring answer"*, and it is why the coverage check has to be **per layer kind against `layer_counts`**,
never over the kinds that happen to have produced output.

Two cells are absent entirely — gemma-4-26B `fuse-noadvise` and llama-3.2-1B `exp17` committed no perf CSV,
so step 0 cannot place them. Treat them as unknown coverage, not good coverage.

**Consequence for v3:** re-derive this table **after** each cell's capture and before screening, from
`reachable_by_advisor` and the reconciliations' `untraced` share. A cell whose dominant kind is still
mostly untraced must publish that number as its headline, not a contribution.

## 2. The run configuration

Recorded here because `~/skillexp-logs/` is not a git repository, so the drivers are otherwise unversioned.
Backups of the originals: `*.bak-preV3`.

| what | v2 | v3 |
|---|---|---|
| `SKILL_BR` | `challenger-skill-v2` | **`challenger-skill-v3`** |
| publish namespace `PUB` | `advchal-v2` | **`advchal-v3`** — v2's tags untouched |
| agent model | **unset**, inherited from the Codex account default | **`--model gpt-5.6-sol`**, explicit (`AGENT_MODEL` overrides) |
| agent effort | unset | **unset, deliberately** |
| incumbent | pinned by SHA per cell | unchanged, and all 11 verified to resolve |

**Why the model is now pinned (A3).** `--model` and `--effort` appeared nowhere in either driver, so v2
inherited the account default. Recovered from the cells' own JSONLs and the driver log: the advchal-v2 cells
ran **`gpt-5.6-sol`** with effort **not set**. An account-default change between v2 and v3 would be silent
and would make every delta read *"stage change **or** agent change"* — which would waste the pinned base
tag the comparison is built on. Effort is left unset on purpose: pinning it to a value v2 did not use
would introduce a new variable rather than remove one.

### The roster, in run order

Order matters: the cell whose outcome is computed in advance runs first, and the negative control runs
early — a control you run last is a control you never acted on.

| # | cell | arm | batch | incumbent | why here |
|---|---|---|---|---|---|
| 1 | `phiFN` | `fuse-noadvise` | 32 | `6e04e475cf41` | **stop-and-reassess.** Must reach −10.43 % |
| 2 | `g26B` | `nofuse-noadvise` | 1 | `3a006fa031dc` | −12.44 % predicted, 26× what it shipped |
| 3 | `g26onA` | `nofuse-noadvise-onA` | 1 | `e578352fc071` | 44 cores over the advised 88 |
| 4 | `nmFN` | `fuse-noadvise` | 1 | `55b77536191d` | 16 cores over the advised 22 and shipped 32 |
| 5 | `nmB` | `nofuse-noadvise` | 1 | `1604664b424a` | coverage: published a flat zero |
| 6 | `nmOnA` | `nofuse-noadvise-onA` | 1 | `ac0f349992f0` | coverage: sparse MoE |
| 7 | `qwenB` | `nofuse-noadvise` | 32 | `ce1b1b13f752` | the 191 ms `retilize`, and the 97 %-of-model kind |
| 8 | `qwen` | `fuse-noadvise` | 32 | `c5c4223d83cb` | the ⚠ row above — its dominant kind is unmeasured |
| 9 | `g26FN` | `fuse-noadvise` | 1 | `851add5b57fa` | no committed CSV; coverage unknown |
| 10–11 | `phiB`, `phiA` | | 32 | | already gained; re-run under the new order |

`run_dense.sh` carries the four snapshot cells — `gemma-4-12B exp11`, `phi exp17`, **`llama-3.1-8B exp17`
(the negative control, which must still report 0.0 %)** and `llama-3.2-1B exp17`. **Run the control
alongside step 1, not at the end.**

**Two cells were missing from the driver and have been added.** In v2, gemma-4-26B `nofuse-noadvise` ended
on a `run/` branch and north-mini `fuse-noadvise` on a `wip/` branch, neither tagged — so neither appeared
in the list the driver replays. They carry **two of the four headline v3 predictions**, so a run without
them could not test them. Incumbents resolved from the surviving arm tags
(`skillexp/done/nofuse-noadvise/google_gemma_4_26b_a4b_it`, `skillexp/done/fuse-noadvise/coherelabs_north_mini_code_1_0`).
This is exactly `FUTURE-RUNS` C4: *a skip must be an artifact, not a log line* — here it was not even a log
line, it was an absence.

## 3. Isolation state of the run clone

Full detail in `~/skillexp-logs/advchal-v3/README.md`, with an offline restore script. Summary: **156 refs
unnamed** (branches, tags, remote-tracking and six deregistered worktree HEADs), `gc.pruneExpire=never` so
no object can be reaped, negative fetch refspecs plus `tagOpt=--no-tags` — verified to hold even against
the driver's own per-cell `git fetch origin --tags`. Step 1's answer is no longer reachable by name or via
`git log --all`.

Residual, recorded rather than closed: two refs stay named because live worktrees hold them
(`publish/phi-exp17`, and the main checkout's own branch), both step-9 cells; and objects remain reachable
by raw SHA, which is acceptable only because the prediction documents are off the skill branch and there is
no in-tree source of a SHA.

## 4. What is still not done

- **The dry run.** `multigoal --dry-run` exercises every step but the model call. Three of four harness bugs
  in the v2 prototype were caught that way in seconds, including a swallowed exit code in the script written
  to prevent swallowed exit codes.
- **A2 in full.** The v2 corpus found five false claims about the advisor in a dependent skill, all biasing
  the agent away from the capability under test. v3's SKILL.md makes new claims — that the five real tracer
  gaps are closed — which are true only for the pinned toolchain; the two-pin check now enforces the
  toolchain, but the prose has not been re-read line by line against it.
- **Exclusivity sampling** is specified in SKILL.md but nothing collects it yet; it belongs in the harness.
