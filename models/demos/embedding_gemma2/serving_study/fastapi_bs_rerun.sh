#!/bin/bash
# FastAPI at batch size 32 (--max-batch 32) vs batch size 1 (template default) and media Python, same window,
# chip 86:00.0 only. Waits for the host load to drop under 4 first. Results: one compare_chain dir per run plus
# fanout and load under it; the summary is grep-able from the log.
source "$(dirname "${BASH_SOURCE[0]}")/env_emb.sh"
[ -n "${EMB_SKIP_LOAD_WAIT:-}" ] || for i in $(seq 120); do l=$(cut -d' ' -f1 /proc/loadavg); awk "BEGIN{exit !($l<4)}" && break; sleep 30; done
echo "load $(cut -d' ' -f1 /proc/loadavg) at $(date -u +%H:%M:%SZ)"
run() { # label, EMB_FASTAPI_ARGS, SRVS for the chain, SRVS for fanout/load
  local L=$E/logs/fastapi_bs_$1_$(date -u +%Y%m%dT%H%M).log
  EMB_FASTAPI_ARGS="$2" SRVS="$3" $EMB_SCRIPTS/compare_chain.sh > $L 2>&1
  local R=$E/$(grep -o "results/[0-9T]*" $L | tail -1)
  EMB_FASTAPI_ARGS="$2" SRVS="$4" $EMB_SCRIPTS/fanout_chain.sh $R >> $L 2>&1
  EMB_FASTAPI_ARGS="$2" SRVS="$4" $EMB_SCRIPTS/load_chain.sh $R >> $L 2>&1
  echo "== $1 ($2) -> $R"; grep -E "^==|^\{|^[a-z_]+ \{|died|fatal|failed|busy" $L | grep -v '"actual_tokens"'
}
export EMB_FASTAPI_ARGS
run bs32 "--max-batch 32" "fastapi" "fastapi"
run bs1 "" "fastapi media_py" "fastapi media_py"
echo "all done $(date -u +%FT%TZ)"; fuser $EMB_DEV 2>/dev/null && echo "chip 31 still held" || echo "chip 31 free"
