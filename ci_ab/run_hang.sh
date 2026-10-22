#!/usr/bin/env bash
# Scratch, not for merge. Hang stress of topk_large_indices on one CI runner. usage: run_hang.sh <sections...>
# sections: suiteW[:passes] suiteN[:passes] glmW[:iters] glmN[:iters] mixW[:iters] mixN[:iters] followW[:iters] followN[:iters]
# W = LLK asserts only (full watcher overflows the kernel config buffer at this head), R = reduced watcher + LLK asserts, N = both off. DEADLINE (s, default 5100) stops starting new work.
set +e +o pipefail
cd /work; export TT_METAL_HOME=/work HWLOCK_HELD=ci
S=/work/ci_ab; OUT=/work/generated/hang; mkdir -p $OUT
T0=$(date +%s); DEADLINE=${DEADLINE:-5100}
echo "== $(date -u +%FT%TZ) host $(hostname) $(git log -1 --format='%h %s' 2>/dev/null)"
(tt-smi -ls 2>&1 | head -8) ; for f in /sys/class/tenstorrent/*/tt_card_type /sys/class/tenstorrent/*/tt_serial; do [ -e "$f" ] && echo "$f $(cat $f)"; done
export TT_METAL_CACHE=$OUT/cache
envW() { unset TT_METAL_WATCHER; export TT_METAL_LLK_ASSERTS=1; }
# R: reduced watcher (NoC sanitize, status polling; waypoints, stack usage, ring buffer, asserts off so the program fits) plus LLK asserts
envR() { export TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_WAYPOINT=1 TT_METAL_WATCHER_DISABLE_STACK_USAGE=1 TT_METAL_WATCHER_DISABLE_RING_BUFFER=1 TT_METAL_WATCHER_DISABLE_ASSERT=1 TT_METAL_LLK_ASSERTS=1; }
envN() { unset TT_METAL_WATCHER TT_METAL_LLK_ASSERTS TT_METAL_WATCHER_DISABLE_WAYPOINT TT_METAL_WATCHER_DISABLE_STACK_USAGE TT_METAL_WATCHER_DISABLE_RING_BUFFER TT_METAL_WATCHER_DISABLE_ASSERT; }
left() { echo $(( DEADLINE - ($(date +%s) - T0) )); }
scan() { grep -a -c -E "TENSIX TIMED OUT|Watcher|watcher.*(assert|error)|NoC|noc.*error|Assertion|TT_THROW|MISMATCH" "$1" 2>/dev/null; }
for sec in "$@"; do
  name=${sec%%:*}; n=${sec#*:}; [ "$n" = "$sec" ] && n=""
  envN; case $name in *W) envW;; *R) envR;; esac
  case $name in
    suite*)
      p=0; tot=${n:-1}
      while [ $p -lt $tot ]; do
        [ $(left) -lt 700 ] && { echo "SECTION $name stop before pass $((p+1)): deadline"; break; }
        p=$((p+1)); t1=$(date +%s); lg=$OUT/${name}_$p.log
        timeout -s INT -k 60 $(( $(left) < 1500 ? $(left) : 1500 )) python3 -m pytest -q -p no:cacheprovider -rfE tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_large_indices.py > $lg 2>&1
        rc=$?; echo "SUITE $name pass $p rc=$rc $(( $(date +%s) - t1 ))s $(grep -a -E '[0-9]+ (passed|failed)' $lg | tail -n 1) hits=$(scan $lg)"
        grep -a -E "^(FAILED|ERROR) " $lg | head -5 | cut -c1-250
        [ $rc -ne 0 ] && { grep -a -E "TENSIX TIMED OUT|TT_THROW|Watcher|MISMATCH" $lg | head -5 | cut -c1-250; }
        [ $rc -ge 124 ] && { echo "SUITE $name pass $p TIMEOUT/KILL, stopping sections that need the device"; break 2; }
      done;;
    glm*|mix*|follow*)
      [ $(left) -lt 300 ] && { echo "SECTION $name skipped: deadline"; continue; }
      mode=glm; X=""; case $name in mix*) X="--mixed";; follow*) mode=follow;; esac
      lg=$OUT/${name}.log
      timeout -s INT -k 60 $(( $(left) )) python3 $S/stress.py $mode --iters ${n:-200} $X --deadline $(( $(left) - 120 )) --seed ${SEED:-7} > $lg 2>&1
      rc=$?; echo "SECTION $name rc=$rc hits=$(scan $lg)"; grep -a -E "^\[.*\] (COUNT|DONE|MISMATCH|mixed probe)" $lg | cut -c1-200
      [ $rc -ne 0 ] && { grep -a -E "TENSIX TIMED OUT|TT_THROW|Watcher|Error|error" $lg | head -8 | cut -c1-250; tail -5 $lg | cut -c1-250; }
      [ $rc -ge 124 ] && { echo "SECTION $name hang or kill"; break; };;
  esac
done
# health: a plain device smoke after everything
envN; timeout -s INT -k 60 300 python3 $S/stress.py smoke > $OUT/health.log 2>&1; echo "HEALTH rc=$? $(tail -1 $OUT/health.log | cut -c1-120)"
echo "== done $(date -u +%FT%TZ) elapsed $(( $(date +%s) - T0 ))s"
