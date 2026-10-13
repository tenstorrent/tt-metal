#!/usr/bin/env bash
# #59149 headroom round, scratch. usage: run_hang3.sh <mode>
#   stop_main:  the id that stopped the bits job (ids_stop.txt) on main's untilize device code, LLK asserts on
#   stop_watch: the same id on this head's code, LLK asserts and the watcher on
#   rest:       the bits ids after it on this head's code with LLK asserts, one process each; then the row-major fp32
#               group norm ids without LLK asserts
set -uo pipefail
R=$(git rev-parse --show-toplevel); S=$R/scratch_hr; cd $R
SP=$(python -c "import ttnn, os; print(os.path.dirname(os.path.dirname(os.path.abspath(ttnn.__file__))))")
echo "== checkout $R head $(git rev-parse HEAD) $(hostname) mode $1"
eval "$(sed -n '/^place() {/,/^}/p' $S/run_hr_op.sh)"
one() {
  local v=$1 lim=$2 id=$3; shift 3
  place $S/variants/$v; export TT_METAL_CACHE=$HOME/hang3cache_$v; mkdir -p $TT_METAL_CACHE
  local t0=$(date +%s)
  env "$@" timeout -s INT -k 20 $lim python -m pytest -q -p no:cacheprovider -rfE "$id" > /tmp/one.log 2>&1
  local rc=$?
  echo "== $v [$*] rc=$rc $(( $(date +%s) - t0 ))s $id :: $(grep -aE '[0-9]+ (passed|failed|skipped|error)' /tmp/one.log | tail -1 | tr -s '=' ' ')"
  if [[ $rc -ne 0 ]]; then grep -aE "^(FAILED|ERROR)|tripped|Watcher|watcher|LLK|assert" /tmp/one.log | head -25 | cut -c1-400; ls -la $R/generated/watcher 2>/dev/null; tail -60 $R/generated/watcher/watcher.log 2>/dev/null | cut -c1-300; fi
}
case $1 in
  stop_main) one main 400 "$(cat $S/ids_stop.txt)" TT_METAL_LLK_ASSERTS=1 ;;
  stop_watch) one fp 400 "$(cat $S/ids_stop.txt)" TT_METAL_LLK_ASSERTS=1 TT_METAL_WATCHER=2 ;;
  rest)
    mapfile -t IDS < $S/ids_after_a.txt
    for f in test_hr_fast_vs_pack.py test_hr_chunk_callers.py; do
      mapfile -t -O ${#IDS[@]} IDS < <(python -m pytest --collect-only -q -p no:cacheprovider $S/tools/$f 2>/dev/null | grep "::")
    done
    echo "== ${#IDS[@]} ids with LLK asserts"
    for id in "${IDS[@]}"; do one fp 300 "$id" TT_METAL_LLK_ASSERTS=1; done
    mapfile -t RM < $S/ids_rmfp32.txt
    for id in "${RM[@]}"; do one fp 300 "$id" -u TT_METAL_LLK_ASSERTS; done ;;
esac
place $S/variants/head
exit 0
