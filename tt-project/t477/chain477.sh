#!/bin/bash
# #480 chain (g15blx02): wait for the blx01 build (build.rc), collect-only check of both test ids (no device),
# then the Tracy profile job (<= 3 tries; cold JIT may need a second), then the pack-SwiGLU A/B job (<= 3 tries).
# One broker job at a time, each through blx01-submit.sh (health + lint + no other smarton job). Marker: chain477.done.
set -eo pipefail
D=$(cd "$(dirname "$0")" && pwd); SUB=/home/smarton/fasth3/tt-metal/tt-project/t288/blx01-submit.sh
O=${CHAIN_OUT:-$D}; M=$O/chain477.done; LOG=$O/chain477.log; rm -f $M
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 g15blx01"; T=/var/tmp/fasth3/t477
trap 'echo $? > "$M"' EXIT
log() { echo "[chain477] $(date -u '+%F %T') $*" | tee -a $LOG; }
for i in $(seq 240); do
  rc=$($SSH cat $T/build.rc 2>/dev/null || true); [ -n "$rc" ] && break; sleep 30
done
log "build rc=${rc:-none}"; [ "$rc" = 0 ] || exit 10
col=$($SSH "cd $T/b && source /var/tmp/fasth3/t460/python_env/bin/activate && export HOME=/var/tmp/fasth3/home TMPDIR=/var/tmp/fasth3/tmp TT_METAL_HOME=$T/b PYTHONPATH=$T/b:$T/b/ttnn:$T/b/tools:/var/tmp/fasth3/t409/pydeps && D=models/tt_dit/tests/models/minimax_h3 && timeout 300 python -m pytest --collect-only -q -p no:cacheprovider \$D/test_transformer_minimax_h3.py::test_minimax_h3_transformer_block_perf \$D/test_ff1_swiglu_pack_minimax_h3.py::test_ff1_swiglu_pack_block_10s -k '4x8sp1tp0nl2_ring_is_fsdp0 and not WH and ((10s_768p and test_prompt_text_tokens and sp_sim1) or block_10s)' 2>&1 | tail -6" || true)
log "collect: $(echo "$col" | tr '\n' '|')"
echo "$col" | grep -q '^2 tests\? collected\|2/[0-9]* tests collected' || exit 15
run() {
  local spec=$1 out id t
  for t in $(seq 40); do
    if out=$($SUB $spec 2>&1); then break; fi
    log "submit $spec refused (try $t): $(echo "$out" | tail -2 | tr '\n' ' ')"; out=; sleep 300
  done
  id=$(echo "$out" | sed -n 's/^job=//p'); [ -n "$id" ] || return 11
  log "$spec -> blx01 job $id"
  while :; do
    s=$($SSH "tt-device-mcp status -j $id" 2>/dev/null | grep '^Status:' || true)
    [ -n "$s" ] && ! echo "$s" | grep -qiE 'running|queued|pending' && break
    sleep 30
  done
  log "job $id: $($SSH "tt-device-mcp status -j $id" 2>/dev/null | grep -iE '^(Status|Exit)' | tr '\n' ' ')"
}
pass() { $SSH "grep -q '^T477_EXIT=0' $T/out_$1/run.log"; }
for t in 1 2 3; do run $D/spec_prof.txt; pass prof && break; done
pass prof || log "profile job did not pass after 3 tries; going on to the A/B"
for t in 1 2 3; do run $D/spec_ab.txt; pass ab && break; done
pass ab || exit 13
pass prof || exit 12
log "all done"
