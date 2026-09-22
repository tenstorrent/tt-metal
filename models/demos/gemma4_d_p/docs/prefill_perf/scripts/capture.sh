#!/usr/bin/env bash
# Parameterised per-op capture + render.
#
#   ./capture.sh --chunk 2048 [--idx 0] [--out DIR] --label NAME [FLAG=VAL ...]
#
# ⚠️ ISOLATED-LAYER capture only. Do NOT point tracy at the full traced run: that is
# ~1.19e9 zones and the OOM killer takes it (and the box) down while saving. Per-layer
# captures are ~3 orders of magnitude smaller. This is why per-op numbers here are
# isolated-layer measurements scaled x50 (sliding) / x10 (global).
#
# --idx picks depth. Use 0 for floor work. To compare chunk WIDTHS at equal prior
# context use idx 28/14/7 for 2048/4096/8192 (= 57,344 tokens). To resolve a SLOPE
# effect use the deepest index instead -- matched-context indices are too shallow, a
# mistake that cost one inconclusive capture pair here.
set -u
T=${TT_METAL_HOME:-/data/kmabee/tt-metal-2}
CHUNK=""; IDX=0; OUT="."; LABEL=""; CTX=256k
declare -a FLAGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --chunk) CHUNK=$2; shift 2;; --idx) IDX=$2; shift 2;;
    --out) OUT=$2; shift 2;;     --label) LABEL=$2; shift 2;;
    --ctx) CTX=$2; shift 2;;
    *=*) FLAGS+=("$1"); shift;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done
[ -n "$CHUNK" ] && [ -n "$LABEL" ] || { sed -n '2,20p' "$0"; exit 1; }

D="$OUT/$LABEL"; mkdir -p "$D"
LOG="$D/run.log"
cd "$T" || exit 1
{ echo "# label $LABEL"; echo "# chunk $CHUNK"; echo "# idx $IDX"
  echo "# git $(git -C "$T" rev-parse --short HEAD)"; echo "# flags ${FLAGS[*]:-none}"; } > "$LOG"

env "${FLAGS[@]}" TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=${PSC:-20000} \
  PYTEST_TIMEOUT=${PYTEST_TIMEOUT:-3600} \
  "$T/python_env/bin/python3" -m tracy -r -p -v -o "$D/profiler" -m pytest \
  "models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunk${IDX}-both-sz${CHUNK}-ctx_${CTX}-8x4]" \
  -sv >> "$LOG" 2>&1
echo "$LABEL: pytest rc=$? -- tracy still post-processing, polling for the ops CSV"

# A capture is NOT done when pytest prints "passed": tracy post-processes for minutes.
CSV=""
for _ in $(seq 1 80); do
  CSV=$(find "$D/profiler" -name 'ops_perf_results_*.csv' -size +1M 2>/dev/null | head -1)
  [ -n "$CSV" ] && break
  sleep 15
done
[ -z "$CSV" ] && { echo "$LABEL: NO CSV -- capture failed"; exit 1; }

for LT in global local; do
  tt-perf-report --start-signpost "gemma4-layer-${LT}-chunk${IDX}-start" \
                 --end-signpost   "gemma4-layer-${LT}-chunk${IDX}-stop" "$CSV" \
                 > "$OUT/${LABEL}_${LT}.txt" 2>&1
  L=$(wc -l < "$OUT/${LABEL}_${LT}.txt")
  [ "$L" -lt 20 ] && echo "  WARNING ${LABEL}_${LT}.txt is only $L lines -- render failed (tt-perf-report on PATH?)"
done
cp "$CSV" "$OUT/${LABEL}_ops.csv"
# ~12 GB per capture; only the ops CSV is needed afterwards, and it is the only thing
# that makes these numbers re-renderable.
find "$D/profiler" -type f \( -name '*.tracy' -o -name 'profile_log_device.csv' \
     -o -name 'tracy_ops_times.csv' -o -name 'tracy_ops_data.csv' \) -delete 2>/dev/null
echo "$LABEL: done, kept $(du -h "$OUT/${LABEL}_ops.csv" | cut -f1) ops CSV"
