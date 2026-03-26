# tt-repos — Manage Parallel tt-metal Repos

A script to run operations across all 5 local `tt-metal-{1..5}` repos in parallel.

**Location:** `.claude/scripts/tt-repos`

**Setup (optional):** Add an alias to `~/.bashrc` so it's callable from anywhere:
```bash
alias tt-repos='/localdev/vignjatijevic/tt-metal-4/.claude/scripts/tt-repos'
```

---

## Commands

### `tt-repos checkout -b`

Creates a new branch in **all 5 repos** simultaneously.

**What it does:**
1. For each `tt-metal-{i}`, checks if any `.claude/` files are git-tracked — if so, untracks them and adds `.claude/` to `.gitignore`.
2. Stages and commits any pending changes (auto-commit message: "Auto-commit before branch switch").
3. Scans local branches matching `vignjatijevic/sfpu-agent-codegen-test-{i}-{j}` to find the highest `j`.
4. Creates and checks out `vignjatijevic/sfpu-agent-codegen-test-{i}-{j+1}`.

**Example:**
```bash
$ tt-repos checkout -b
Creating next branch in all repos...
  [tt-metal-1] Committed pending changes
  [tt-metal-1] → vignjatijevic/sfpu-agent-codegen-test-1-2
  [tt-metal-2] → vignjatijevic/sfpu-agent-codegen-test-2-2
  [tt-metal-3] → vignjatijevic/sfpu-agent-codegen-test-3-2
  [tt-metal-4] → vignjatijevic/sfpu-agent-codegen-test-4-2
  [tt-metal-5] → vignjatijevic/sfpu-agent-codegen-test-5-2
```

---

### `tt-repos -D <pattern>`

Deletes a branch across **all 5 repos** in parallel. Use `{i}` in the pattern as a placeholder for the repo index — the script prepends `vignjatijevic/` automatically.

**What it does:**
1. For each `tt-metal-{i}`, replaces `{i}` in the pattern with the repo index to form the full branch name `vignjatijevic/<pattern>`.
2. If the repo is currently on that branch, checks out to the base branch (`vignjatijevic/sfpu-agent-codegen-test-{i}`) first.
3. Force-deletes the branch (`git branch -D`).
4. If the branch doesn't exist in a repo, it is skipped with a warning.

**Examples:**
```bash
# Delete vignjatijevic/sfpu-agent-codegen-{i}-1 across all repos:
$ tt-repos -D sfpu-agent-codegen-{i}-1
Deleting branches matching vignjatijevic/sfpu-agent-codegen-{i}-1 ...
  [tt-metal-1] Checked out to vignjatijevic/sfpu-agent-codegen-test-1
  [tt-metal-1] Deleted vignjatijevic/sfpu-agent-codegen-1-1
  [tt-metal-2] Checked out to vignjatijevic/sfpu-agent-codegen-test-2
  [tt-metal-2] Deleted vignjatijevic/sfpu-agent-codegen-2-1
  [tt-metal-3] Checked out to vignjatijevic/sfpu-agent-codegen-test-3
  [tt-metal-3] Deleted vignjatijevic/sfpu-agent-codegen-3-1
  [tt-metal-4] Checked out to vignjatijevic/sfpu-agent-codegen-test-4
  [tt-metal-4] Deleted vignjatijevic/sfpu-agent-codegen-4-1
  [tt-metal-5] Checked out to vignjatijevic/sfpu-agent-codegen-test-5
  [tt-metal-5] Deleted vignjatijevic/sfpu-agent-codegen-5-1

# Delete a specific numbered iteration:
$ tt-repos -D sfpu-agent-codegen-test-{i}-3
Deleting branches matching vignjatijevic/sfpu-agent-codegen-test-{i}-3 ...
  [tt-metal-1] Deleted vignjatijevic/sfpu-agent-codegen-test-1-3
  [tt-metal-2] Branch vignjatijevic/sfpu-agent-codegen-test-2-3 does not exist, skipping
  ...
```

**Note:** The pattern is everything after `vignjatijevic/`. Make sure to quote the pattern or escape `{i}` if your shell expands braces (e.g. `tt-repos -D 'sfpu-agent-codegen-{i}-1'`).

---

### `tt-repos sync <path> [<dst>...]`

Copies a folder from the repo you're currently in to other repos. The target folder is **deleted and replaced** entirely.

**Arguments:**
- `<path>` — relative folder path within the repo (e.g. `ttnn/cpp/ttnn/operations/eltwise/unary`)
- `[<dst>...]` — optional destination repos. Accepts:
  - Bare index: `3`
  - Full name: `tt-metal-3`
  - If omitted, syncs to **all** other repos.

**Examples:**
```bash
# From inside tt-metal-4, sync a folder to ALL other repos:
$ tt-repos sync ttnn/cpp/ttnn/operations/eltwise/unary
Syncing 'ttnn/cpp/ttnn/operations/eltwise/unary' from tt-metal-4 → tt-metal-1, tt-metal-2, tt-metal-3, tt-metal-5...
  [tt-metal-1] Synced ttnn/cpp/ttnn/operations/eltwise/unary
  [tt-metal-2] Synced ttnn/cpp/ttnn/operations/eltwise/unary
  [tt-metal-3] Synced ttnn/cpp/ttnn/operations/eltwise/unary
  [tt-metal-5] Synced ttnn/cpp/ttnn/operations/eltwise/unary

# Sync to specific repos only:
$ tt-repos sync .claude/scripts 2 5
Syncing '.claude/scripts' from tt-metal-4 → tt-metal-2, tt-metal-5...
  [tt-metal-2] Synced .claude/scripts
  [tt-metal-5] Synced .claude/scripts

# Using full repo name:
$ tt-repos sync some/path tt-metal-3
Syncing 'some/path' from tt-metal-4 → tt-metal-3...
  [tt-metal-3] Synced some/path
```

**Note:** The source repo is auto-detected from your current working directory. If you specify the source repo as a destination, it is silently skipped.

---

### `tt-repos status`

Shows the current branch and working tree state for all repos.

**Example:**
```bash
$ tt-repos status
Repository status:
  [tt-metal-1] vignjatijevic/sfpu-agent-codegen-test-1-1
  [tt-metal-2] vignjatijevic/sfpu-agent-codegen-test-2-1 (dirty) (+3 untracked)
  [tt-metal-3] vignjatijevic/sfpu-agent-codegen-test-3-1
  [tt-metal-4] vignjatijevic/sfpu-agent-codegen-test-4-1 (dirty)
  [tt-metal-5] vignjatijevic/sfpu-agent-codegen-test-5-1
```

- **(dirty)** — there are uncommitted staged or unstaged changes
- **(+N untracked)** — there are N untracked files

---

### `tt-repos exec <command>`

Runs an arbitrary shell command inside every repo in parallel. Output is grouped per repo.

**Examples:**
```bash
# Check recent commits across all repos:
$ tt-repos exec "git log --oneline -3"

# See what branch each repo is on:
$ tt-repos exec "git branch --show-current"

# Run a grep across all repos:
$ tt-repos exec "grep -r 'SFPU_OP_CHAIN' --include='*.cpp' -l | head -5"
```

---

### `tt-repos commit "<message>"`

Stages all changes and commits in every repo with the given message. Automatically untracks `.claude/` files before committing (same as `checkout -b`).

**Example:**
```bash
$ tt-repos commit "Add new SFPU operation: elu"
Committing across all repos...
  [tt-metal-1] Committed: Add new SFPU operation: elu
  [tt-metal-2] Nothing to commit
  [tt-metal-3] Committed: Add new SFPU operation: elu
  [tt-metal-4] Committed: Add new SFPU operation: elu
  [tt-metal-5] Nothing to commit
```

---

### `tt-repos branches`

Lists all local branches matching the codegen naming pattern for each repo.

**Example:**
```bash
$ tt-repos branches
Codegen branches per repo:
  [tt-metal-1]
      vignjatijevic/sfpu-agent-codegen-test-1
      vignjatijevic/sfpu-agent-codegen-test-1-1
      vignjatijevic/sfpu-agent-codegen-test-1-2
  [tt-metal-2]
      vignjatijevic/sfpu-agent-codegen-test-2
      vignjatijevic/sfpu-agent-codegen-test-2-1
  ...
```

---

## Configuration

The following variables at the top of the script can be adjusted:

| Variable | Default | Purpose |
|---|---|---|
| `BASE_DIR` | `/localdev/vignjatijevic` | Parent directory containing all repos |
| `REPO_PREFIX` | `tt-metal` | Repo directory prefix |
| `REPO_INDICES` | `(1 2 3 4 5)` | Which repo indices to manage |
| `BRANCH_PREFIX` | `vignjatijevic/sfpu-agent-codegen-test` | Branch naming prefix for `checkout -b` |

---

## Keeping the script in sync

Since `.claude/` is gitignored, the script lives only locally. After editing it in one repo, sync it to the others:

```bash
tt-repos sync .claude/scripts
```
