#!/usr/bin/env bash
# Fast decode-perf gate (~1 min): warm ISL 4096 (1 pass) then measure ONE point (ISL 4096 / OSL 64,
# conc 1) -> ms/token + t/s/u + TTFT. Appends a one-line result to activity.log. Arg1 = label.
# Optional: set QP_ISL / QP_OSL to override the measured point.
set +e
LABEL="${1:-probe}"
ISL="${QP_ISL:-4096}"; OSL="${QP_OSL:-64}"
BASE=/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1
ACT=$BASE/doc/vllm_integration/activity.log
OUTD=$BASE/doc/vllm_integration/smoke/quickperf; mkdir -p "$OUTD"
VBIN=/home/ttuser/.tenstorrent-venv/bin
export PYTHONPATH=/home/ttuser/dev/tt-metal:/home/ttuser/.local/lib/model-bringup/tt-metal/vllm:/home/ttuser/.local/lib/model-bringup/tt-metal/vllm/plugins/vllm-tt-plugin/src
bench(){ timeout 300 "$VBIN/vllm" bench serve --backend vllm --base-url http://localhost:8000 --endpoint /v1/completions \
  --model poolside/Laguna-XS-2.1 --dataset-name random --num-prompts 1 --random-input-len "$ISL" --random-output-len "$1" \
  --max-concurrency 1 --ignore-eos --save-result --result-filename "$2" > "$2.log" 2>&1; }
echo "   [quick_perf $(date +%T)] $LABEL: warming ISL=$ISL…" >> "$ACT"
bench 16 "$OUTD/warm.json"
bench "$OSL" "$OUTD/m.json"; rc=$?
python3 - "$OUTD/m.json" "$LABEL" "$ISL" "$OSL" "$rc" >> "$ACT" <<'PY'
import json,sys
f,lab,isl,osl,rc=sys.argv[1:6]
if rc!="0":
    print(f"   [quick_perf] {lab}: FAIL (rc={rc}) ISL={isl} OSL={osl}"); sys.exit()
try:
    d=json.load(open(f)); tpot=d["mean_tpot_ms"]; ttft=d["mean_ttft_ms"]
    print(f"   [quick_perf] {lab}: ISL={isl} OSL={osl}  ms/token={tpot:.1f}  t/s/u={1000/tpot:.1f}  TTFT={ttft:.0f}ms")
except Exception as e:
    print(f"   [quick_perf] {lab}: parse-fail {e}")
PY
