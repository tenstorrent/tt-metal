#!/bin/bash
# fmt_commit.sh <git commit args...>: format staged files (black / clang-format, repo configs), re-stage, commit --no-verify
# (pre-commit can't install its envs on this box).
P=/home/mmanzoor/compiler/tt-metal/python_env/bin
files=$(git diff --cached --name-only --diff-filter=AM)
for f in $files; do
  case $f in
    *.py) $P/black -q -l 120 "$f" 2>/dev/null ;;
    *.cpp|*.hpp|*.h|*.cc) $P/clang-format -i --style=file "$f" ;;
  esac
done
git add $files
git commit --no-verify "$@"
