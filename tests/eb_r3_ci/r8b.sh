#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass, final code (#58725): main's program against the head with eight sections for two operand
# passes and the Python scalar and the L1 step-down, three passes; and which bf16 and bfp8 height shards allocate, each case in
# a fresh process, main against the head, with the failure lines.
cd /work
export EB_R3_LOG_RULE=1
M="EB_R3_NO_BLOCK=1 EB_R3_NO_BCAST_CHUNK=1 EB_R3_MAIN_REINIT=1 EB_R3_PER_FACE=1 EB_R3_NO_HIFI3=1 EB_R3_NO_PRE_SECTIONS=1 EB_R3_NO_NATIVE=1"
for i in 1 2 3; do
  echo "##### pass $i k8m: main vs head"; bash tests/eb_r3_ci/ab_envs.sh "$M" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_r5.py -k "test_k8"
done
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
IDS=$(python3 -c "print(' '.join(f'{op}-t{t}' for op in ('logical_and','rsub_s','rsub') for t in (128,144,160,176,192,208,224,240)))")
for side in main head; do
  for t in $IDS; do
    if [[ $side == main ]]; then E="$M"; else E="EB_DUMMY=1"; fi
    r=$(env $E timeout 600 python3 -m pytest -p no:cacheprovider -q -rfE tests/eb_r3_ci/test_eb_r5.py -k "test_l1probe and $t" 2>&1)
    o=$(echo "$r" | grep -E "passed|failed|error" | tail -1 | cut -c1-60); w=$(echo "$r" | grep -E "^E |TT_FATAL|TT_THROW|clash|EB_R3_RULE l1" | head -3 | cut -c1-220 | tr '\n' ' ')
    echo "L1PROBE $side $t: $o $w"
  done
done
echo "##### l1bits: main vs head"; bash tests/eb_r3_ci/bits_envs.sh "$M" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_r5.py -k "test_l1probe"
