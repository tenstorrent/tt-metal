#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Phase 2.2 A/B measurement: what does ttnvtop-collector cost a running
# workload, and how much of that cost is specifically the remote-chip
# ethernet tunnel?  (PLAN_ETH_AGGREGATOR.md §5)
#
# Three arms, so the remote cost is isolated rather than inferred:
#
#   A       collector off                          -> baseline
#   B_local collector on, LOCAL chips only          -> cost of collector at all
#   B_all   collector on, ALL chips incl. remote    -> baseline + tunnel cost
#
#   (B_all - B_local) is the remote-tunnel cost. That is the number that
#   decides whether Phase 2.2.a is worth building.
#
# Arms are interleaved and rotated per rep, not run in blocks, so thermal
# drift and AICLK changes cannot masquerade as an effect.
#
# Usage (on the target host, from TT_METAL_HOME):
#   bash tt_metal/tools/ttnvtop/scripts/ab_remote_cost.sh --reps 5
#
# Env overrides: LOCAL_CHIPS, ALL_CHIPS, COLLECTOR_BIN, SAMPLE_HZ, PUBLISH_HZ

set -uo pipefail

TT_METAL_HOME="${TT_METAL_HOME:-$(pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR_BIN="${COLLECTOR_BIN:-${TT_METAL_HOME}/build_Release/tools/ttnvtop-collector}"
WORKLOAD_PY="${WORKLOAD_PY:-${SCRIPT_DIR}/ab_workload.py}"

REPS=5
ITERS=50000
SIZE=2048
WARMUP=500
MESH_ROWS=2
MESH_COLS=4
# T3K: chips 0-3 are MMIO/local, 4-7 are remote. Verify on the box before trusting.
LOCAL_CHIPS="${LOCAL_CHIPS:-0,1,2,3}"
ALL_CHIPS="${ALL_CHIPS:-0,1,2,3,4,5,6,7}"
SAMPLE_HZ="${SAMPLE_HZ:-300}"
PUBLISH_HZ="${PUBLISH_HZ:-100}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reps)   REPS="$2"; shift 2 ;;
    --iters)  ITERS="$2"; shift 2 ;;
    --size)   SIZE="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --rows)   MESH_ROWS="$2"; shift 2 ;;
    --cols)   MESH_COLS="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${TT_METAL_HOME}/runs/ab-${TS}"
mkdir -p "$OUT_DIR"
RESULTS="${OUT_DIR}/results.jsonl"

echo "=== Phase 2.2 A/B — remote-chip collector cost ==="
echo "host      : $(hostname)"
echo "out       : ${OUT_DIR}"
echo "workload  : ${MESH_ROWS}x${MESH_COLS} mesh, ${SIZE}^2 matmul, ${ITERS} iters (+${WARMUP} warmup)"
echo "reps      : ${REPS} per arm, interleaved"
echo

[[ -x "$COLLECTOR_BIN" ]] || { echo "FATAL: no collector at ${COLLECTOR_BIN}" >&2; exit 1; }
[[ -f "$WORKLOAD_PY"  ]] || { echo "FATAL: no workload at ${WORKLOAD_PY}" >&2; exit 1; }

COLLECTOR_PID=""

start_collector() {  # $1 = comma-separated chip list, $2 = log path
  # --device takes a single int and must be REPEATED (collector/main.cpp:252),
  # so expand the comma list into one flag per chip.
  local dev_args=() chip
  IFS=',' read -ra _chips <<< "$1"
  for chip in "${_chips[@]}"; do dev_args+=(--device "$chip"); done

  "$COLLECTOR_BIN" "${dev_args[@]}" --sample-hz "$SAMPLE_HZ" --publish-hz "$PUBLISH_HZ" \
      >"$2" 2>&1 &
  COLLECTOR_PID=$!
  # let topology discovery + counter arming settle before the workload starts
  sleep 8
  if ! kill -0 "$COLLECTOR_PID" 2>/dev/null; then
    echo "  WARN: collector exited early; see $2" >&2
    COLLECTOR_PID=""
    return 1
  fi
  return 0
}

stop_collector() {
  [[ -n "$COLLECTOR_PID" ]] || return 0
  kill -INT "$COLLECTOR_PID" 2>/dev/null
  wait "$COLLECTOR_PID" 2>/dev/null
  COLLECTOR_PID=""
  sleep 2
}

cleanup() { stop_collector; }
trap cleanup EXIT INT TERM

run_arm() {  # $1 = arm name, $2 = rep index
  local arm="$1" rep="$2"
  local log="${OUT_DIR}/${arm}-rep${rep}"
  local clog="${log}.collector.log"

  case "$arm" in
    A)       : ;;
    B_local) start_collector "$LOCAL_CHIPS" "$clog" || return 1 ;;
    B_all)   start_collector "$ALL_CHIPS"   "$clog" || return 1 ;;
  esac

  local out
  out="$(python3 "$WORKLOAD_PY" --rows "$MESH_ROWS" --cols "$MESH_COLS" \
          --size "$SIZE" --iters "$ITERS" --warmup "$WARMUP" \
          --label "${arm}" 2>"${log}.stderr")"
  local rc=$?
  echo "$out" > "${log}.stdout"

  stop_collector

  local json
  json="$(grep -o 'TTNVTOP_AB_RESULT .*' <<<"$out" | sed 's/^TTNVTOP_AB_RESULT //')"
  if [[ $rc -ne 0 || -z "$json" ]]; then
    echo "  ${arm} rep${rep}: FAILED (rc=${rc}) — see ${log}.stderr" >&2
    return 1
  fi

  python3 - "$json" "$arm" "$rep" "$RESULTS" <<'PY'
import json, sys
d = json.loads(sys.argv[1]); d["arm"] = sys.argv[2]; d["rep"] = int(sys.argv[3])
with open(sys.argv[4], "a") as f: f.write(json.dumps(d) + "\n")
print(f"  {d['arm']:<8} rep{d['rep']}  {d['elapsed_s']:7.3f} s   "
      f"{d['per_iter_ms']:6.2f} ms/iter   {d['tflops']:6.1f} TFLOP/s")
PY
}

# Rotate arm order every rep so a monotonic drift (thermal, clock) spreads
# evenly across arms instead of loading onto whichever runs last.
ARMS=(A B_local B_all)
for ((r = 1; r <= REPS; r++)); do
  echo "--- rep ${r}/${REPS} ---"
  for ((i = 0; i < ${#ARMS[@]}; i++)); do
    idx=$(( (i + r - 1) % ${#ARMS[@]} ))
    run_arm "${ARMS[$idx]}" "$r"
  done
done

echo
echo "=== summary ==="
python3 - "$RESULTS" "$OUT_DIR" <<'PY'
import json, statistics as st, sys, glob, os, re
rows = [json.loads(l) for l in open(sys.argv[1])]
out_dir = sys.argv[2]
by = {}
for r in rows: by.setdefault(r["arm"], []).append(r["elapsed_s"])

def stat(v):
    m = st.mean(v)
    s = st.stdev(v) if len(v) > 1 else 0.0
    return m, s

if "A" not in by:
    print("no baseline runs succeeded"); sys.exit(1)
base, base_sd = stat(by["A"])
print(f"{'arm':<9} {'n':>2} {'mean_s':>9} {'sd_s':>7} {'vs A':>9}")
for arm in ("A", "B_local", "B_all"):
    if arm not in by: continue
    m, s = stat(by[arm])
    d = (m - base) / base * 100.0
    print(f"{arm:<9} {len(by[arm]):>2} {m:>9.3f} {s:>7.3f} {d:>8.2f}%")

if "B_all" in by and "B_local" in by:
    ma, _ = stat(by["B_all"]); ml, _ = stat(by["B_local"])
    tunnel = (ma - ml) / base * 100.0
    print(f"\nremote-tunnel cost (B_all - B_local) = {ma-ml:+.3f} s = {tunnel:+.2f}% of baseline")
    noise = max(base_sd, 1e-9) / base * 100.0
    print(f"baseline run-to-run noise (sd)       = {noise:.2f}%")
    print()
    if abs(tunnel) < 2 * noise:
        print(">>> Tunnel cost is WITHIN 2x baseline noise.")
        print(">>> Phase 2.2.a (transport) is NOT justified by workload impact.")
        print(">>> Build 2.2.b only, justified by sample fidelity.")
    else:
        print(f">>> Tunnel cost is {tunnel:.2f}%, above noise. Phase 2.2.a is justified.")

# Collector-side drain stats, for the fidelity half of the argument.
print("\n--- collector drain (last line per run) ---")
for f in sorted(glob.glob(os.path.join(out_dir, "*.collector.log"))):
    lines = [l for l in open(f, errors="replace") if "[ring-drain]" in l]
    if lines:
        print(f"  {os.path.basename(f):<28} {lines[-1].strip()[:110]}")
PY

echo
echo "artifacts: ${OUT_DIR}"
