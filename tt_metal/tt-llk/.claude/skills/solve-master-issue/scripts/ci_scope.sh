#!/bin/bash
# usage: ci_scope.sh --pr <n> | <worktree> | --files <path>...
# Classifies a PR's change set for §6: prints `ttllk-only` when every changed file is under one of the
# LLK-only prefixes (default tt_metal/tt-llk/ — the tt-llk headers and their pytest suite, which Sanity's
# LLK smoke and the LLK e2e workflow run on silicon), else `full` followed by up to 5 of the other paths.
# Policy (Anil, 2026-10-17): a ttllk-only PR is gated by Sanity + LLK e2e; the Nightly L2 is not dispatched
# for it (dispatch.sh enforces this). CI_SCOPE_LLK_PREFIXES="p1 p2" overrides the prefix list.
# Exit 0 ttllk-only · 1 full · 2 usage/error.
set -uo pipefail
prefixes=(${CI_SCOPE_LLK_PREFIXES:-tt_metal/tt-llk/})
files=()
case ${1:-} in
--pr)   [ -n "${2:-}" ] || { echo "usage: ci_scope.sh --pr <n>" >&2; exit 2; }
        mapfile -t files < <(gh pr view "$2" --repo "${GH_REPO:-tenstorrent/tt-metal}" --json files -q '.files[].path') || exit 2 ;;
--files) shift; files=("$@") ;;
"")     echo "usage: ci_scope.sh --pr <n> | <worktree> | --files <path>..." >&2; exit 2 ;;
*)      base=$(git -C "$1" merge-base HEAD origin/main) || exit 2
        mapfile -t files < <(git -C "$1" diff --name-only "$base" HEAD) || exit 2 ;;
esac
[ ${#files[@]} -gt 0 ] || { echo "full (no files found)"; exit 1; }
other=()
for f in "${files[@]}"; do
    ok=0; for p in "${prefixes[@]}"; do [[ $f == "$p"* ]] && { ok=1; break; }; done
    [ $ok = 1 ] || other+=("$f")
done
if [ ${#other[@]} -eq 0 ]; then echo ttllk-only; exit 0; fi
echo "full ${other[*]:0:5}"; exit 1
