#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass (#58725): which bf16 height shards allocate with the operand pass per section (main's
# program), over four sections (the head) and over eight; each case in a fresh process so no earlier test fragments L1.
cd /work
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
IDS=$(python3 -c "print(' '.join(f'{op}-t{t}' for op in ('logical_and','rsub_s','rsub') for t in (128,144,160,176,192,208,224,240)))")
for side in "EB_R3_NO_PRE_SECTIONS=1" "EB_R3_NONE=1" "EB_R3_PRE_MAX=8"; do
  for t in $IDS; do
    r=$(env $side timeout 600 python3 -m pytest -p no:cacheprovider -q -rfE tests/eb_r3_ci/test_eb_r5.py -k "test_l1probe and $t" 2>&1)
    o=$(echo "$r" | grep -E "passed|failed|error" | tail -1 | cut -c1-80); w=$(echo "$r" | grep -o "bank_manager.cpp:[0-9]*\|Out of Memory[^.]*" | head -1)
    echo "L1PROBE $side $t: $o $w"
  done
done
