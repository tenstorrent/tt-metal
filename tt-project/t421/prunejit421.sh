#!/bin/bash
# t421: prune blx01 JIT kernel cache entries whose source tree no longer exists (their hash includes
# the tree path, so they can never hit again). Usage: prunejit421.sh [--delete]. Prints KiB per tree.
set -eo pipefail
K=/var/tmp/fasth3/cache/tt-metal-cache-h3hf/tt-metal-cache9268051462127607463/kernels
declare -A kb n
for h in "$K"/*/*/; do
  f=$(find "$h" -maxdepth 2 -name '*.dephash' -print -quit)
  [ -n "$f" ] || continue
  t=$(grep -m1 -oa '/[a-z0-9_./-]*/tt_metal/' "$f" | head -1 | sed 's#/tt_metal/$##')
  [ -n "$t" ] || continue
  [ -d "$t/tt_metal" ] && continue
  s=$(du -x -sk "$h" | cut -f1); kb[$t]=$(( ${kb[$t]:-0} + s )); n[$t]=$(( ${n[$t]:-0} + 1 ))
  [ "$1" = --delete ] && rm -rf -- "$h"
done
for t in "${!kb[@]}"; do echo "${kb[$t]} KiB ${n[$t]} entries $t"; done | sort -n
