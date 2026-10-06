# t48 cleanup plan: move notes and tmp/ off ttp/t48-ltx25-integrated (#146, NOT run)

Written 2026-10-06 against origin/ttp/t48-ltx25-integrated @ b81bb403d86 (2026-10-06 03:47 UTC).
Nothing here has been run. No step rewrites or force-pushes t48. Step 4 (PR branch) waits for the
user's word on PRs.

## What is on t48 that should not be

Base for the counts: merge-base(t48, origin/ltx-rt) = b9f8587ce6c6. That range has 318 commits,
15 of them merges, and touches 198 files.

Non-code paths in the t48 tree (38 files):
- `NOTES.md` (top level, t48 integration notes)
- `tmp/READY_22.md`, `tmp/READY_26.md`, `tmp/READY_48.md`
- `tmp/block_env.yaml`, `tmp/chunk_ab.sh`, `tmp/cmp.sh`, `tmp/cmp_blk.py`, `tmp/done.sh`,
  `tmp/drive6.sh`, `tmp/e2e.sh`, `tmp/na_kernel_cmds.txt`, `tmp/na_kernel_compile.sh`
- `tmp/blx03/`: `NOTES.md`, `READY_blx03.md`, `drv_template/{driver.sh,test_health.sh}`, `env.yaml`,
  `run25.sh`, `run48.sh`, `submit.sh` (**shared infra**, see Risks)
- `tmp/t60/` (7 files), `tmp/t138/` (3), `tmp/t140/` (4), `tmp/t141/` (3)
- `tt-project/t93/PLAN.md`

Commits that add them:
- 47 notes-only commits (only those paths): `notes_only_commits.txt` next to this file. They span
  t10, t20, t22, t26, t36, t48, t60, t93, t138, t140 and t141, from 09-30 to 10-06.
- 7 mixed commits (code plus notes). Their code must stay:
  - ee87923c0f2 open_trace_gate / crf 23: `NOTES.md`, `tmp/READY_48.md`
  - 558c15e0b97 fold T pad default: `tmp/READY_48.md`, `tmp/blx03/run48.sh`
  - e21bcbad8d9 neighbor_pad logical_w: `tmp/t60/kernel_compile.sh`, `tmp/t60/np_kernel_cmds.txt`
  - 0e1a2585b17 LTX_VAE_FOLD_W_MASK: five `tmp/t60/*` files
  - f5a920523b2 t60 shallow submodules: `tmp/t60/blx03_setup60.sh`
  - 81df0effdd4 READY_48 own build: `tmp/READY_48.md`, `tmp/blx03/run48.sh`, `tmp/t60/blx03_setup60.sh`
  - 7e25dc0dbad s4_res/s1_up blockings: `NOTES.md`
- Merges that bring notes in from task branches (t8, t13, t18, t22, t44) carry the same files.

Task branches on origin hold older copies of the same notes with different hashes (e.g.
ttp/t140-eval-pack-4x8 @ ef7f918402c vs t48's t140 commits). `ttp push` rebased each task's notes
onto t48, which made new hashes. That is the copy problem the guard now stops.

## What stops it from growing (done in #146)

`delivery.push_checks` now starts with `tt-project/harness/checks/push-branch-code-only.sh`. A
`ttp push` to t48 fails (exit 4) when any new commit adds or changes `tmp/`, `tt-project/`,
`NOTES.md` or `READY_*.md`. Deletions pass, so the step-3 commit can land. `ttp push --own` and
`ttp checks` are not affected. `prompts/kind-code.md` ("Notes stay off the push branch") tells
workers how to split code from notes.

## Method: forward-only, built next to t48

History rewrite (filter-repo, or rebase -i with drops) is rejected. The range has 15 merges, and every
open task worktree and origin branch is based on today's hashes. A rewrite needs a force-push of t48
and makes every running task's next `ttp push` conflict.

### Step 0. Preconditions (check, don't force)
- No push in flight: `tt-project/harness/bin/ttp push --free` exits 0.
- No task is mid-way through a t48 push or depends on t48's `tmp/` files on g15blx02. Today that
  means #140 (t140 driver), #141 and #136 have finished, and #127 and #135 are not in the middle of a
  push. Their blx03 drivers run from copies on blx03 (`~/fasth3/t*drv/`, `/var/tmp/fasth3/t140/src`),
  so they don't depend on t48.
- The coordinator has scheduled the cleanup as its own task.

### Step 1. Archive (lossless, no rewrite)
- `git push origin origin/ttp/t48-ltx25-integrated:refs/heads/ttp/t48-notes-archive`
  (a new branch name, created with a plain push, no force). Every note and driver stays
  reachable there, with the hashes it has today.
- Write `tt-project/t48-cleanup/ARCHIVE.md` with the archive sha and the file list above.

### Step 2. Rehome the shared blx03 scripts first
- `tmp/blx03/{submit.sh,env.yaml,drv_template/*,run25.sh,run48.sh}` are shared tooling, not task
  notes. Copy them into the harness (`tt-project/harness/templates/blx03/`). Line them up with #145's
  serial blx03 runner (`templates/blx03-runner/`) so there is one copy. Point `blx03-launch.sh`'s
  comment at the new path. It now says "copy of tmp/blx03/drv_template/driver.sh (branch with 29a0e8dfdc8)".
- Do not touch blx03's `~/fasth3/tt-metal`. It is on `ttp/t36-blx03-ltx25` @ 86076afc66 with local
  edits to `tmp/blx03/NOTES.md` and `run25.sh`, and running drivers call its `tmp/blx03/submit.sh`
  and `env.yaml`. It never reads t48 and stays as it is.

### Step 3. One forward removal commit on t48 (fast-forward, via `ttp push`)
- In a fresh task worktree off origin/t48:
  `git rm -r -q tmp NOTES.md tt-project && git commit -m "ltx: move task notes and tmp/ drivers off the integration branch"`
  with the body naming the archive branch and sha.
- Verify before pushing:
  - `git diff --name-only origin/ttp/t48-ltx25-integrated HEAD` lists only the 38 paths above.
  - `git diff origin/ttp/t48-ltx25-integrated HEAD -- . ':!tmp' ':!tt-project' ':!NOTES.md'` is empty.
- Land with `ttp push --detach`. The guard lets deletions through, and the pytest checks run on the
  result. This is an ordinary fast-forward, not a rewrite, and needs no force. Check with the user
  first anyway, because it changes the shared branch every task rebases onto.

### Step 4. PR branch, only when the user allows PRs (built next to t48, t48 untouched)
- `git switch -c ttp/t48-pr origin/ltx-rt` (or whatever base the user names for the PR), then
  `git merge --squash origin/ttp/t48-ltx25-integrated`. Split the result into a few logical
  commits (conv3d/neighbor_pad C++, LTX VAE, denoise/pipeline, tests) with
  `git restore --staged` / `git add <paths>`.
- If step 3 has not landed, first drop the paths with
  `git rm -r --cached tmp NOTES.md tt-project`.
- Verify the trees: `git diff origin/ttp/t48-ltx25-integrated ttp/t48-pr -- . ':!tmp' ':!tt-project' ':!NOTES.md'`
  must be empty, and `git ls-tree -r --name-only ttp/t48-pr | grep -E '^(tmp|tt-project)/|NOTES.md'`
  must print nothing.
- Run `ttp checks` on it, then `ttp push --own` once the branch name matches the task's own name.
  The PR is a draft, and only after the user says so.
- The PR's diff has no notes. The notes stay in the archive and task branches.

## Risks for running and queued tasks

- Rebase conflicts: a task branch that changes a file step 3 deletes (e.g. `tmp/t140/*`,
  `tmp/blx03/run48.sh`) gets a modify/delete conflict (exit 3) on its next `ttp push`. The guard
  already refuses such pushes, so the fix is the same: split code from notes (kind-code.md) and land
  only the code. Run step 3 when #140, #141, #136, #127 and #135 have no push pending.
- Drivers that read t48's tree: none found on g15blx02. On blx03, drivers read
  `~/fasth3/tt-metal/tmp/blx03/*` (the t36 checkout) and their own copies. Step 3 does not change
  either. A future `git checkout` of t48 in blx03's tree would drop `tmp/blx03/submit.sh`, so do
  step 2 first and point drivers at the harness copy.
- `templates/blx03-launch.sh` and older NOTES refer to `tmp/blx03/drv_template/driver.sh` on
  t48. After step 3 they must use the step-2 copy or the archive branch.
- Concurrent pushes: step 3 goes through `ttp push`, which holds the push lock and rebases. A push
  that lands between the verify and the push is handled by its rounds. Re-run the verify on the
  final head (the push log shows it).
- Lost context: none. Everything removed stays in `ttp/t48-notes-archive`, in task branches and in
  the root's `tt-project/t*` folders.
- Guard false positives: a model change that really needs a file named `NOTES.md` or under `tmp/` is
  refused. None exists on ltx-rt today. If one is ever needed, change the guard's pattern in the
  harness. Never bypass it per push.
