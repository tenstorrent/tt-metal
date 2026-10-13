# Knowledge staleness re-check

Re-verify the bring-up knowledge entries whose code changed since they were last checked, and fix or remove the ones
that are no longer true. Run it after a rebase of the bring-up branch, after a fork sync (`ttnn/ttnn/bringup`), or when
the branch's origin moved under work you depend on. It is self-contained: this file is all you need.

## What is checked

- The shared files, read by every bring-up agent: `models/demos/common/bringup/knowledge/known_issues.md` (sections,
  one bullet per lesson: `Symptom:`, `Cause:`, `Fix:`, `Applies to:`, `Seen at:`, optional `Check:`, `Found:`) and
  `repo_map.md` (a `| Need | Where |` table).
- Each model's own files, read by that model's agents: `models/demos/<model>/bringup/known_issues.md` (bullets with
  `Symptom:`, `Cause:`, `Fix:`, `Found:`; `Applies to:` / `Seen at:` optional) and `bringup/repo_map.md`.
- `Applies to:` names the code an entry is about (ops, forks, repo paths); `Seen at:` is the commit at which it was
  last known true. An entry whose `Applies to:` code has commits after `Seen at:` may be stale.

## Rules

- No device runs (no pytest on device, no `scripts/run_safe_pytest.sh`, no `scripts/tt-probe.sh`). You decide by
  reading code and git history. What only a device run can settle is marked, not guessed.
- Change only the knowledge files above. Do not commit unless the person who started you said so.
- Every change needs evidence you can cite: a commit (`git log`) or a `file:line` in the current tree.
- Do not touch entries whose code did not change, and do not add new lessons from unrelated reading.
- Keep each entry's `Found:` (who found it, in which task). Keep the format: run the checker (step 5) before finishing.

## 1. List the candidates

From the repo root, with `PYTHONPATH=$PWD` and the repo's python env active:

```
python -m models.demos.common.bringup.knowledge.stale --out <scratch dir>
```

It writes `stale.md` (one row per entry, sorted by status) and `stale.json` (per entry: `file`, `line`, `title`,
`text`, `since` (its `Seen at:`, else the commit that last touched it), `anchors` (the repo paths its `Applies to:`
resolved to), `commits` (since..HEAD on those paths), `missing`, `status`). Statuses:

| Status | Meaning | What you do |
|---|---|---|
| `changed` | its anchored code has commits after `since` | verify it (step 2) |
| `anchor-missing` | a path it names existed at `since` and is gone | find where the code went (step 2) |
| `unanchored` | nothing it names resolved to a path | judge it from its text (step 3) |
| `unchanged` | no commits on its code since `since` | nothing |

It also lists repo-map rows naming paths that no longer exist: fix the path or drop the row.

## 2. Verify each `changed` / `anchor-missing` entry

1. Read the entry. Name the one claim it makes (an op refuses X, an option defaults to Y, a workaround is needed
   because Z).
2. Look only at the commits listed for it: `git log --stat <since>..HEAD -- <anchors>`, then
   `git show <commit> -- <file>` for the commits whose subject or files touch the claim. Most listed commits are
   unrelated (cleanups, other features): skip them after reading the subject and stat.
3. Read the current code where the claim lives (the validation, the default, the kernel path) and decide:

| Verdict | When | Edit |
|---|---|---|
| still true | the claim holds in the current code | set `Seen at:` to the current HEAD (short sha); nothing else |
| changed | the claim holds partly, or the fix / workaround changed (a new option, a renamed path, a fork now covers it) | rewrite the smallest part that changed; `Seen at:` = HEAD |
| fixed | the problem is gone (the op accepts X now, the default changed) and the lesson has no general value left | delete the entry |
| moved | `anchor-missing`: the code moved (`git log --follow`, `git log -S '<symbol>'`) | update `Applies to:` (and the text if it names the path); then judge it as above |
| needs a device run | the claim is about numbers or behaviour the code does not settle (precision, a hang, a perf figure) | keep the text; add or keep a `Check:` describing the probe that would settle it; add "unverified since `<HEAD>`" before `Found:` |

For a model's file, also add `Applies to:` / `Seen at:` to an entry you verified, so the next re-check can follow it.

## 3. Judge each `unanchored` entry

Read its text. If it is about code, find that code (grep the symbol, the op, the test helper), add an `Applies to:`
that resolves (a repo path, or an op's Python name such as `unified_routed_expert_moe`), and verify it as in step 2.
If it is method or process knowledge (how to test, how to read a metric), it cannot go stale with the code: leave it.

## 4. Repo maps

For each row of the shared and per-model `repo_map.md`: a path that no longer exists is fixed (where did it go?) or
the row is dropped; a row pointing at an original op that now has a `ttnn.bringup` fork the models use names the fork
too. Do not restructure rows that are still right.

## 5. Check and report

```
python -m models.demos.common.bringup.knowledge.check                         # the shared files
python -m models.demos.common.bringup.knowledge.check --model-dir models/demos/<model>   # each model you touched
```

Both must exit 0 (sections, fields, anchors and the shared length cap of 800 characters per entry). Then report, in a
file next to `stale.md` (`report.md`) and briefly in your final message:
- counts: entries checked, still true, changed, fixed (deleted), moved, needs a device run, unanchored judged;
- one line per edited or deleted entry: `file:line`, title, verdict, the evidence (commit or `file:line`);
- the "needs a device run" list with the probe for each;
- anything you could not decide.
