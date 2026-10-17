#!/bin/bash
# usage: mkwt.sh <short-name> <branch>
# Worktree at ${WORKTREE_ROOT:-<parent of repo>}/wt-<short-name> on <branch>: a new branch off
# origin/main, or the existing branch (local or origin/) when resuming. The tt-llk test sfpi/.venv
# directories are symlinked from the main checkout when present — never stage those symlinks.
# Run from inside the repository (or set REPO_ROOT).
set -euo pipefail
main=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
name=${1#wt-}   # "wt-foo" and "foo" both mean wt-foo (a run once created wt-wt-foo)
wt=${WORKTREE_ROOT:-$(dirname "$main")}/wt-$name
[ -e "$wt" ] && { echo "$wt already exists" >&2; exit 1; }
mkdir -p "$(dirname "$wt")"
free_gb=$(df -BG --output=avail "$(dirname "$wt")" | tail -1 | tr -dc 0-9)
[ "${free_gb:-0}" -lt 20 ] && echo "warning: only ${free_gb}G free (a checkout + Release build is ~8G, plus ~2G harness)" >&2
git -C "$main" fetch -q origin main "$2" 2>/dev/null || git -C "$main" fetch -q origin main
if git -C "$main" show-ref -q --verify "refs/heads/$2"; then
    git -C "$main" worktree add -q "$wt" "$2"
elif git -C "$main" show-ref -q --verify "refs/remotes/origin/$2"; then
    git -C "$main" worktree add -q -b "$2" "$wt" "origin/$2"
else
    git -C "$main" worktree add -q -b "$2" "$wt" origin/main
fi
t=tt_metal/tt-llk/tests
for l in sfpi .venv; do
    src=$(readlink -f "$main/$t/$l" 2>/dev/null || true)
    [ -n "$src" ] && [ -e "$src" ] && ln -s "$src" "$wt/$t/$l"
done
echo "$wt"
