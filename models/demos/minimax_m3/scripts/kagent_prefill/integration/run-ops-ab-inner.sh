#!/bin/bash
rc=0
NEW=/mnt/data/kernel-agent/dev/prefill-best/tt-metal
OLD=/mnt/data/kernel-agent/dev/prefill/tt-metal
while read -r label tree rest; do
  [ -z "$label" ] && continue; [[ "$label" == \#* ]] && continue
  set -- $rest
  envs=(); while [[ "$1" == *=* ]]; do envs+=("$1"); shift; done
  script=$1; shift
  if [ "$tree" = old ]; then T=$OLD; C=$KP/jit-cache-old; else T=$NEW; C=$KP/jit-cache; fi
  echo "=== CASE $label ($tree: $T) ${envs[*]} $script $*"
  (cd $T && env TT_METAL_HOME=$T TT_METAL_RUNTIME_ROOT=$T PYTHONPATH=$T:$T/ttnn:$T/tools TT_METAL_CACHE=$C "${envs[@]}" \
     python -u $NEW/models/demos/minimax_m3/tests/perf/$script "$@" < /dev/null); r=$?
  echo "=== CASE $label rc=$r"
  [ $r -ne 0 ] && rc=$r
done < "$1"
exit $rc
