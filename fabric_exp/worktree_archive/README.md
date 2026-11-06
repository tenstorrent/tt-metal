# Worktree archive (2026-11-06)

Before the study's worktrees under /home/aagarwal were removed:

- `<worktree>/HEAD.txt`: the commit and branch the worktree had checked out, and where that commit lives (a remote
  branch, or a local tag `archive/<worktree>` in the study repo).
- `<worktree>/uncommitted.diff`: `git diff HEAD --binary` of its uncommitted changes (apply with `git apply` on that commit).
- `<worktree>/untracked/`: its untracked text files under 2 MB (build trees, generated/, python_env and binaries skipped;
  `untracked_skipped.txt` lists the larger ones).
- `archive_heads.bundle`: every `archive/*` tag's commits not already on origin/main or the router branches, so the
  heads survive without the local repo: `git fetch archive_heads.bundle 'refs/tags/archive/*:refs/tags/archive/*'`.
