#!/bin/bash
# D5 (c): the R1-2f variance table. Full bench (both precisions, all shapes) for the base (every role on
# ttnn.linear), R1-2f, and R1-2f with one role group reverted at a time. Run under ttq --kind bench from the
# worktree root; writes <out>/<variant>.json.
set -u
out=${1:?out dir}
mkdir -p "$out"
declare -A V=(
  [base]="embed,head,mlstm,ffn,w_in,variate"
  [r12f]=""
  [rev_mlstm]="mlstm"
  [rev_ffn_down]="ffn"
  [rev_w_in]="w_in"
  [rev_embed_head]="embed,head"
  [rev_variate]="variate"
)
rc=0
for v in base r12f rev_mlstm rev_ffn_down rev_w_in rev_embed_head rev_variate; do
  echo "=== $v (TIREX2_MM_REVERT=${V[$v]})"
  TIREX2_MM_REVERT="${V[$v]}" MODEL_PERF_JSON="$out/$v.json" \
    python -m pytest -q models/experimental/tirex2_forecast/tests/perf/test_forward_perf.py -s 2>&1 | grep -E "passed|failed|Error" || rc=1
done
exit $rc
