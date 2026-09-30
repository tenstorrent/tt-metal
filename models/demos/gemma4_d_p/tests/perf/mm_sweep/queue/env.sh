# Sasha follow-ups (from 2026-09-30): tree /data/kmabee/tt-metal-3 (tt-metal-2 is in use from another machine).
export W=/data/kmabee/tt-metal-3
source /data/kmabee/runs_0927/common.sh      # sources runs_0925/lib.sh: perf, pcc, waitchips, mkb, purge, t
O=/data/kmabee/runs_sasha
# perfx <label> <chunk> <ctx_id e.g. 256k> [ENV=V...]
perfx () { local L=perf_$1_c$2_$3; local C=$2 X=$3; shift 3
  waitchips; echo "=== $L start $(date +%T) sha=$(git -C $W rev-parse --short HEAD) env=$*"
  (cd $W && env "$@" TT_METAL_HOME=$W PYTHONPATH=$W/ttnn:$W timeout 2400 $PY -m pytest "models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_long_context_traced[blackhole-readback_final-ctx_${X}-chunk${C}-text-8x4]" -sv -p no:cacheprovider < /dev/null > $O/$L.log 2>&1)
  echo "$L rc=$? $(python3 $O/tt.py $O/$L.log)"; }
tdp () { tt-smi -s 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('tdp', sorted({x.get('limits',{}).get('tdp_limit') for x in d['device_info']}))"; }
# Per-tree, per-commit JIT cache: the build key does not include the tree or commit, so tt-metal-2 and
# tt-metal-3 would otherwise share kernel binaries. Uncommitted kernel-header edits still need a purge.
export TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-t3/$(git -C $W rev-parse --short HEAD)
# waitchips override: the lib version never returns when a dead process left locks in files owned by
# another user (rm is not permitted, so they stay "claimed" forever). Proceed when every claim is STALE
# (owner dead), after one glx_reset; wait while any claim has a live owner.
waitchips () {
  while true; do
    local s; s=$(~/scripts/tt-devs.sh 2>/dev/null)
    echo "$s" | grep -q " 0/32 chips claimed" && return
    local live; live=$(echo "$s" | grep -E "^/dev/tenstorrent/" | grep -v -E "free|STALE" | wc -l)
    if [ "$live" = 0 ]; then echo "stale-only locks -> glx_reset $(date +%T)"; tt-smi -glx_reset > $O/glx_reset_auto.log 2>&1; return; fi
    sleep 20
  done; }
