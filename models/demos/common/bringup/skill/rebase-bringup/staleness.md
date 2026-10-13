# Knowledge staleness re-check

Re-verify the bring-up knowledge entries whose code changed since they were last checked, and fix or remove the ones
that are no longer true. Run it after a rebase of the bring-up branch, after a fork sync (`ttnn/ttnn/bringup`), or when
the branch's origin moved under work you depend on. It is self-contained: this file is all you need.

## What is checked

- The shared files, read by every bring-up agent: `models/demos/common/bringup/knowledge/known_issues.md` (sections,
  one bullet per lesson) and `repo_map.md` (a `| Need | Where |` table).
- Each model's own files, read by that model's agents: `models/demos/<model>/bringup/known_issues.md` (bullets, no
  sections, at most 600 characters each) and `bringup/repo_map.md`. A bullet whose Fix ends with
  `Detail: known_issues_detail.md.` has its full text in that file under `## <title>`: when you rewrite or delete the
  bullet, update or delete that section too.
- An entry is one bullet, `- **[<id>] <title>.**` (the id never changes: keep it when you rewrite an entry, and never
  reuse a deleted one), with these fields, in this order: `Symptom:`, `Cause:`, `Fix:`, `Applies to:` (the code it is
  about: repo paths, partial paths such as `tt/mhc.py`, op names such as `unified_routed_expert_moe`; or the word
  `method` for testing / process knowledge that cannot go stale with the code), `Seen at:` (the commit at which it was
  last known true), optionally one `Check:` (the probe that settles it on a device), optionally
  `Unverified since: <commit>.` (a device check is still open), `Found:` (who found it, in which task; never changed).
  Shared entries require `Applies to:` and `Seen at:` and stay within 800 characters; model entries get them when you
  verify them.

## Rules

- Change only the knowledge files above. Do not commit unless the person who started you said so.
- Every change needs evidence you can cite: a commit (`git log`), a `file:line` in the current tree, or a device result.
- Do not touch entries whose code did not change, and do not add new lessons from unrelated reading.
- Device runs only in step 4, and only through `scripts/tt-probe.sh` or `scripts/run_safe_pytest.sh`, in the
  foreground, one at a time. If the person said no device runs, skip step 4 and leave its entries marked.

## 1. List the candidates

From the repo root, with `PYTHONPATH=$PWD` and the repo's python env active:

```
python -m models.demos.common.bringup.knowledge.stale --out <scratch dir>
```

It writes `stale.md` (one row per entry, sorted by status) and `stale.json` (per entry: `file`, `line`, `title`,
`text`, `since`, `anchors` (the repo paths its `Applies to:` resolved to), `commits` (since..HEAD on those paths),
`older_commits`, `missing`, `status`). A rebase reorders history, so commits written before `since` can show up after
it; the lister drops them (`older_commits`) and keeps the rest.

| Status | Meaning | Step |
|---|---|---|
| `needs-probe` | carries `Unverified since:`: a device check is open | 4 |
| `contradicted` | an agent's knowledge-use report (after `since`) says the code or a device result disagrees; `contradictions` holds the evidence | 2 |
| `anchor-missing` | a path it names existed at `since` and is gone | 2 |
| `changed` | its anchored code has commits after `since` | 2 |
| `unanchored` | nothing it names resolved to a path | 3 |
| `method` | `Applies to: method` | nothing |
| `unchanged` | no commits on its code since `since` | nothing |

It also lists repo-map rows naming paths that no longer exist (step 5).

More than ~30 candidates: split them by file (shared, then one model per worker), each worker editing only its own
files, and merge; two workers never edit the same file.

## 2. Verify each `contradicted` / `changed` / `anchor-missing` entry

A `contradicted` entry: start from the reports' evidence (`contradictions` in stale.json: task, commit, the agent's
words), check it against the current code, and decide as below. An agent can be wrong too: "still true" with the reason
is a valid verdict.

1. Name the one claim the entry makes (an op refuses X, an option defaults to Y, a workaround is needed because Z, a
   measured number).
2. Look only at its listed commits: `git log --stat <since>..HEAD -- <anchors>`, then `git show <commit> -- <file>` for
   those whose subject or files touch the claim. Most are unrelated (cleanups, other features): skip them after the
   subject and stat.
3. Read the current code where the claim lives and decide:

| Verdict | When | Edit |
|---|---|---|
| still true | the claim holds in the current code | `Seen at:` = HEAD (short sha); nothing else |
| changed | any part changed: a detail is wrong, a path moved, a fork or a new option now gives a better fix | rewrite the smallest part (a new option goes first in `Fix:`); `Seen at:` = HEAD |
| fixed | the problem is gone and the lesson has no general value left | delete the entry |
| moved | `anchor-missing`: the code moved (`git log --follow`, `git log -S '<symbol>'`) | update `Applies to:` (and the text); then judge it as above |
| needs a device run | the claim is a measured number or a behaviour the code does not settle (precision, a hang, a timing), and the commits touch what produces it (the kernel, the precision, the chunking, the collective), or the claim was measured on a configuration that has since changed | keep the text; one `Check:` naming the probe and the mesh it needs; `Unverified since: <HEAD>.`; keep `Seen at:` (step 4 settles it) |

Unchanged code around a measured number is enough to keep it (still true) only when no listed commit touches what
produces the number. When the fix is a model setting, `Applies to:` names the file that reads it (the model's
`tt/settings.py`, `tt/model.py`, ...), so the next re-check follows it. A model entry you verify gets `Applies to:` and
`Seen at:` before `Found:`.

## 3. Judge each `unanchored` entry

If it is about code, find that code (grep the symbol, the op, the test helper), give it an `Applies to:` that resolves
(a repo path, or an op's Python name), and verify it as in step 2. If it is testing or process knowledge, set
`Applies to: method` and leave the rest.

## 4. Device checks

For every entry marked "needs a device run" (in step 2, or already `needs-probe`):
1. Read its `Check:`. If this box cannot open the mesh it needs (`ls /dev/tenstorrent | wc -l`, the model spec's
   `box.mesh`), leave the entry as it is and list it in the report with the box it needs.
2. Write the smallest probe that answers the claim: a standalone script (random inputs, the op or the model test with
   the switch the entry names) run with `scripts/tt-probe.sh <name> < probe.py`, or an existing test / profile through
   `scripts/run_safe_pytest.sh` with the entry's settings. Bound it in time (a watchdog for a claimed hang). Print the
   numbers the entry states, next to the reference they are measured against.
3. Record the result:
   - confirmed: put the measured numbers in the text if they changed, `Seen at:` = HEAD, drop `Unverified since:`;
     keep `Check:` (the next re-check can rerun it);
   - refuted (the problem is gone): delete the entry, or rewrite it if part still holds;
   - inconclusive: keep `Unverified since:`, and say in the report what the probe showed.
4. Keep the probe in `models/demos/common/bringup/knowledge/probes/<name>.py` (with the SPDX header and its run
   line) and name it in the entry's `Check:`, so the next re-check reruns the same probe. tt-probe's own copy under
   `tests/ttnn/unit_tests/operations/<name>/probes/` is scratch: delete it.

## 5. Repo maps

For each row of the shared and per-model `repo_map.md`: a path that no longer exists is fixed (where did it go?) or the
row is dropped. When you rewrite an entry about an op or option, grep both repo maps for that op and update rows that
now say something wrong about it (a fork that now covers it, a new option). Do not restructure rows that are still
right.

## 6. Check and report

```
python -m models.demos.common.bringup.knowledge.check                                  # the shared files
python -m models.demos.common.bringup.knowledge.check --model-dir models/demos/<model>   # each model you touched
```

Both must exit 0. Then rerun step 1: what is left `changed` must be entries you decided on with a reason. Report, in
`report.md` next to `stale.md` and briefly in your final message:
- counts: entries checked, still true, changed, fixed (deleted), moved, device checks run (confirmed / refuted /
  inconclusive), device checks left open (with the box each needs), unanchored judged;
- one line per edited or deleted entry: `file:line`, title, verdict, the evidence (commit, `file:line`, or probe);
- anything you could not decide.
