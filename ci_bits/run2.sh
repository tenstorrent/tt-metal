#!/usr/bin/env bash
# #59149 headroom round, scratch: the untilize callers the face-pair passes reach. LLK asserts on for every caller but the
# sharded group norm, whose Welford fp32 configs trip main's unpack format assert (cunpack_common.h unp_A_dst_format, on
# main's code too); the group norm ids run after, without LLK asserts.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); cd $R
export PYTHONPATH=$R${PYTHONPATH:+:$PYTHONPATH}
echo "== head $(git rev-parse HEAD) $(hostname)"
mapfile -t A < ci_bits/ids_a.txt
TT_METAL_LLK_ASSERTS=1 timeout -s INT -k 60 3000 python -m pytest -q -p no:cacheprovider -o timeout_method=thread -rfE "${A[@]}" > /tmp/a.log 2>&1
echo "== callers, LLK asserts on, rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed)' /tmp/a.log | tail -1)"
grep -aE "^(FAILED|ERROR) " /tmp/a.log | cut -c1-300
mapfile -t G < ci_bits/ids_gn.txt
timeout -s INT -k 60 2400 python -m pytest -q -p no:cacheprovider -o timeout_method=thread -rfE "${G[@]}" > /tmp/g.log 2>&1
echo "== group norm, rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed)' /tmp/g.log | tail -1)"
grep -aE "^(FAILED|ERROR) " /tmp/g.log | cut -c1-300
exit 0
