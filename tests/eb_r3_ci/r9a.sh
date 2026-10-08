#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass, merged head (#58725, #58726): the sections set at invoke from L1. A filler leaves less
# L1 free, one process from roomy to tight, main's program against the head (outcomes and bits), then the head's plan per
# case; and the probe of 13:05 again, each case in a fresh process, main against the head.
cd /work
M="EB_R3_NO_BLOCK=1 EB_R3_NO_BCAST_CHUNK=1 EB_R3_MAIN_REINIT=1 EB_R3_PER_FACE=1 EB_R3_NO_HIFI3=1 EB_R3_NO_PRE_SECTIONS=1 EB_R3_NO_NATIVE=1"
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
echo "##### l1fill bits: main vs head"; bash tests/eb_r3_ci/bits_envs.sh "$M" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_r5.py -k "test_l1fill"
echo "##### l1fill head plan"
EB_R3_LOG_RULE=1 timeout 1800 python3 -m pytest -p no:cacheprovider -v -s -rfE tests/eb_r3_ci/test_eb_r5.py -k "test_l1fill" 2>&1 \
  | grep -E "test_l1fill\[|EB_R3_RULE l1|PASSED|FAILED|TT_FATAL|TT_THROW|clash" | sed -E 's/\x1b\[[0-9;]*m//g' | cut -c1-260
echo "##### l1fill main outcomes"
env $M timeout 1800 python3 -m pytest -p no:cacheprovider -q -rfE tests/eb_r3_ci/test_eb_r5.py -k "test_l1fill" 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed" | cut -c1-200
IDS=$(python3 -c "print(' '.join(f'{op}-t{t}' for op in ('logical_and','rsub_s','rsub') for t in (128,144,160,176,192,208,224,240)))")
for side in main head; do
  for t in $IDS; do
    if [[ $side == main ]]; then E="$M"; else E="EB_DUMMY=1"; fi
    r=$(env $E EB_R3_LOG_RULE=1 timeout 600 python3 -m pytest -p no:cacheprovider -q -rfE tests/eb_r3_ci/test_eb_r5.py -k "test_l1probe and $t" 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g')
    o=$(echo "$r" | grep -E "passed|failed|error" | tail -1 | cut -c1-60); w=$(echo "$r" | grep -E "TT_FATAL|TT_THROW|EB_R3_RULE l1" | sort -u | head -3 | cut -c1-200 | tr '\n' ' ')
    echo "L1PROBE $side $t: $o $w"
  done
done
echo "##### l1probe bits: main vs head"; bash tests/eb_r3_ci/bits_envs.sh "$M" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_r5.py -k "test_l1probe"
