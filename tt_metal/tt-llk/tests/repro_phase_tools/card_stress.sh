#!/bin/bash
cd /home/nstojic/tt-metal && git fetch -q origin nstojictt/p58-versim && git -C /tmp/p58 checkout -q --detach origin/nstojictt/p58-versim && git -C /tmp/p58 log --oneline -1
# card checks for the M1c, T5, T6, T7 Versim choices; results in /tmp/cq/<name>/
out=/tmp/adv; mkdir -p $out
export CHIP_ARCH=wormhole USER=gitlab-ci
run() { # name tree env... -- ids...
  n=$1; tree=$2; shift 2; envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  [ -s $out/$n.log ] && grep -q "passed\|failed" $out/$n.log && { echo "$n done before"; return; }
  ( cd $tree/tt_metal/tt-llk/tests/python_tests && source ../.venv/bin/activate && \
    env LLK_HOME=$tree/tt_metal/tt-llk RUNNER_TEMP=/tmp/build-adv-$n NNG_SOCKET_NAME=adv "${envs[@]}" python -u -m pytest -p no:randomly -q "$@" > $out/$n.log 2>&1 )
  for f in $(grep -o "Wrote run Parquet batch: [^ ]*" $out/$n.log | awk '{print $NF}' | sort -u); do mkdir -p $out/$n; cp $(dirname $f)/*/*.csv $out/$n/ 2>/dev/null; done
  echo "$n $(tail -n1 $out/$n.log)"
}
grep -v "perf_eltwise_unary_sfpu.py\|perf_eltwise_binary_sfpu.py\|perf_eltwise_unary_typecast.py\|perf_sfpu_ternary.py\|perf_vif_targets.py" /tmp/adv_set.txt > $out/set2.txt; mapfile -t IDS < $out/set2.txt; tac $out/set2.txt > $out/set2_rev.txt; mapfile -t REV < $out/set2_rev.txt; T=/tmp/p58
run um1 $T LLK_FN_NOPS_THREADS=UNPACK,MATH LLK_THREAD_FN_NOPS=1 -- "${IDS[@]}"
run s2k $T LLK_ISO_SETTLE=2000 -- "${IDS[@]}"
run s2k_um1 $T LLK_ISO_SETTLE=2000 LLK_FN_NOPS_THREADS=UNPACK,MATH LLK_THREAD_FN_NOPS=1 -- "${IDS[@]}"
run pk1 $T LLK_FN_NOPS_THREADS=PACK LLK_THREAD_FN_NOPS=1 -- "${IDS[@]}"
run s2k_pk1 $T LLK_ISO_SETTLE=2000 LLK_FN_NOPS_THREADS=PACK LLK_THREAD_FN_NOPS=1 -- "${IDS[@]}"
run nowarm $T LLK_PERF_NO_WARMUP=1 -- "${IDS[@]}"
run nowarm_rev $T LLK_PERF_NO_WARMUP=1 -- "${REV[@]}"
run zr1 $T LLK_ZONE_RESERVE_NOPS=1 -- "${IDS[@]}"
run bn1 $T LLK_BRISC_FN_NOPS=1 -- "${IDS[@]}"
echo ADVDONE
