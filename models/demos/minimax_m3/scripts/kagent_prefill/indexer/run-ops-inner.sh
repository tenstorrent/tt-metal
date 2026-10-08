#!/bin/bash
# inner loop of run-ops.sh (already inside the partition job, cwd = worktree)
rc=0
while read -r label rest; do
  [ -z "$label" ] && continue; [[ "$label" == \#* ]] && continue
  envs=(); set -- $rest
  while [[ "$1" == *=* ]]; do envs+=("$1"); shift; done
  script=$1; shift
  extra=()
  for e in "${envs[@]}"; do
    if [ "$e" = "BASE=1" ]; then
      extra+=(TT_METAL_HOME=$KP/base-root TT_METAL_RUNTIME_ROOT=$KP/base-root TT_METAL_CACHE=$KP/jit-cache-base)
    else extra+=("$e"); fi
  done
  echo "=== CASE $label: ${envs[*]} $script $*"
  env "${extra[@]}" python -u models/demos/minimax_m3/tests/perf/$script "$@" < /dev/null; r=$?
  echo "=== CASE $label rc=$r"
  [ $r -ne 0 ] && rc=$r
done < "$1"
exit $rc
