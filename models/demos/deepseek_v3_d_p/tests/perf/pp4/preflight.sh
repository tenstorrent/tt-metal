#!/usr/bin/env bash
# Preflight for the Mistral4 PP=4 vs single-rank perf work on a NEW machine.
#
# Several things this work depends on are machine-specific in ways that fail SILENTLY or produce
# wrong numbers rather than erroring. Run this first on any new box and read every FAIL/WARN.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
cd "$TT_METAL_HOME" || { echo "FAIL: checkout $TT_METAL_HOME not reachable"; exit 1; }
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export LD_LIBRARY_PATH=$PWD/build_Release/lib:${LD_LIBRARY_PATH:-}
PY=./python_env/bin/python
ok(){ echo "  OK   $*"; }; warn(){ echo "  WARN $*"; }; fail(){ echo "  FAIL $*"; RC=1; }
RC=0

echo "== host =="
echo "  $(hostname)  |  branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"

echo "== chips =="
N=$(ls /dev/tenstorrent 2>/dev/null | grep -c '^[0-9]*$')
echo "  /dev/tenstorrent devices: $N"
if [ "$N" -eq 32 ]; then ok "32 chips -> PP=4 x (8,1) is possible"
else fail "PP=4 x (8,1) needs a 32-chip galaxy; this box has $N. The [8,1] bindings CANNOT run here."; fi

echo "== chips actually free (other users' processes are invisible to fuser) =="
BUSY=0; for d in /dev/tenstorrent/*; do [ -n "$(fuser "$d" 2>/dev/null)" ] && BUSY=$((BUSY+1)); done
[ "$BUSY" -eq 0 ] && ok "no processes of MINE hold devices" || warn "$BUSY device(s) held by my own procs"
OTHER=$(ps -eo user,etime,pcpu,cmd --no-headers | awk '$1!="'"$USER"'" && $3>50 {print}' | grep -iE "pytest|python|tt-|vllm" | head -3)
[ -n "$OTHER" ] && warn "another user may hold the chips:
$OTHER" || ok "no other user's heavy job visible"

echo "== build =="
[ -f build_Release/lib/_ttnn.so ] && ok "_ttnn.so present" || fail "build_Release missing"
grep -q "^ENABLE_TRACY:BOOL=ON" build_Release/CMakeCache.txt 2>/dev/null \
  && ok "ENABLE_TRACY=ON (device profiler usable)" || fail "build lacks ENABLE_TRACY -> no Tracy/per-op capture"
A=$(grep -oE "^ARCH_NAME:[A-Z]*=.*" build_Release/CMakeCache.txt 2>/dev/null | head -1)
echo "  ${A:-ARCH_NAME not in cache}"

echo "== tools =="
command -v tt-perf-report >/dev/null && ok "tt-perf-report: $(command -v tt-perf-report)" \
  || warn "tt-perf-report NOT on PATH (it lives in ~/.local/bin, which is LOCAL disk per machine; pip install it here)"
command -v tt-smi >/dev/null && ok "tt-smi present" || warn "tt-smi missing (needed to reset after a hard kill)"

echo "== shared data (NFS /data - should all be present) =="
for p in "$MISTRAL4_HF_MODEL" \
         "$M4_CACHE_8x4/mistral_small4_bh_32dev/8x4" \
         "$M4_CACHE_8x1/mistral_small4_bh_8dev/8x1" \
         "$GOLDEN_5120"; do
  [ -e "$p" ] && ok "$(basename "$p")" || fail "missing $p"
done
L=$(ls ${M4_CACHE_8x1}/mistral_small4_bh_8dev/8x1 2>/dev/null | grep -oE '^layer_[0-9]+' | sed 's/layer_//' | sort -un | wc -l)
[ "$L" = "36" ] && ok "per-stage 8x1 cache has all 36 global layers" || fail "8x1 cache has $L layers, need 36 (all PP ranks share one dir, keyed by GLOBAL layer index)"

echo "== per-host, will be COLD here =="
warn "TT_METAL_CACHE (/tmp/tt-metal-cache-pp) is per-host: first run of each shape pays full kernel JIT (+1-2 min)"
[ -d "$HOME/debug-docs" ] && ok "~/debug-docs present" \
  || warn "~/debug-docs ABSENT (\$HOME is local disk). Docs for this work are on shared ${TT_METAL_HOME}/models/demos/deepseek_v3_d_p/tests/perf/pp4/"

echo "== CRITICAL: the [8,1] column -> physical device map is per-galaxy =="
echo "  The rank bindings hardcode TT_VISIBLE_DEVICES read off bh-glx-b03u02's 8x4 mesh."
echo "  A different galaxy may enumerate differently, and a WRONG map does not error - it silently"
echo "  builds a pipeline whose stages are not the columns you think. Re-derive and compare:"
echo "     \$PY scripts/probe_columns.py   # needs the galaxy idle"
echo "  Expected on bh-glx-b03u02:"
echo "     rank0 0,1,2,3,11,10,9,8   rank1 4,5,6,7,15,14,13,12"
echo "     rank2 28,29,30,31,23,22,21,20   rank3 24,25,26,27,19,18,17,16"

echo
[ "$RC" = "0" ] && echo "PREFLIGHT: no FAILs (read the WARNs)" || echo "PREFLIGHT: FAILures above - fix before running"
exit $RC
