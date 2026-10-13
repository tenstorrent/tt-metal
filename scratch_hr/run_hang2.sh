#!/usr/bin/env bash
# #59149 headroom round, scratch: find the test that stopped run_hang.sh. Each id of the rest set in its own process on
# this head's device code with LLK asserts on; an id that times out runs again on the head's code (when it builds there)
# and with the watcher, so a tripped assert names itself.
set -uo pipefail
R=$(git rev-parse --show-toplevel); S=$R/scratch_hr; cd $R
SP=$(python -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== checkout $R head $(git rev-parse HEAD) $(hostname)"
eval "$(sed -n '/^place() {/,/^}/p' $S/run_hr_op.sh)"
one() {
  local v=$1 lim=$2 id=$3 extra=${4:-}
  place $S/variants/$v; export TT_METAL_CACHE=$HOME/hang2cache_$v; mkdir -p $TT_METAL_CACHE
  local t0=$(date +%s)
  env TT_METAL_LLK_ASSERTS=1 $extra timeout -s INT -k 20 $lim python -m pytest -q -p no:cacheprovider -rfE "$id" > /tmp/one.log 2>&1
  local rc=$?
  echo "== $v rc=$rc $(( $(date +%s) - t0 ))s $id :: $(grep -aE '[0-9]+ (passed|failed|skipped|error)' /tmp/one.log | tail -1 | tr -s '=' ' ')"
  if [[ $rc -ne 0 ]]; then grep -aE "^(FAILED|ERROR)|tripped|assert|Watcher|WATCHER|LLK" /tmp/one.log | grep -av "^$" | head -15 | cut -c1-300; fi
  return $rc
}
mapfile -t IDS < $S/ids_rest2.txt
for f in test_hr_fast_vs_pack.py test_hr_chunk_callers.py; do
  mapfile -t -O ${#IDS[@]} IDS < <(python -m pytest --collect-only -q -p no:cacheprovider $S/tools/$f 2>/dev/null | grep "::")
done
echo "== ${#IDS[@]} ids"
for id in "${IDS[@]}"; do
  one fp 300 "$id"; rc=$?
  if [[ $rc -eq 124 || $rc -eq 137 || $rc -eq 130 ]]; then
    [[ $id != *scratch_hr/tools/test_hr_fast_vs_pack* ]] && one head 300 "$id"
    one fp 300 "$id" "TT_METAL_WATCHER=5"
  fi
done
place $S/variants/head
exit 0
