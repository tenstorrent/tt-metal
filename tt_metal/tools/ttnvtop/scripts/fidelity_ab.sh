#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# THE fidelity measurement (PLAN_ETH_AGGREGATOR.md 5s).
#
# This is the only surviving justification for on-chip aggregation. The original one --
# "monitoring stalls workloads" -- did not reproduce: three Llama-70B arms spanned 0.5%
# with the ordering backwards (5l/5q). So the question this script answers is narrow and
# decisive: over ONE workload run, does the host per-core drain lose samples that the
# on-chip sweep does not?
#
# Both consumers read the same 62-entry per-core rings and NEITHER consumes, so they run
# CONCURRENTLY against one workload. That is deliberate. Separate runs would leave "were
# the two arms even producing the same samples" as a variable, and sample production
# depends on the workload's kernel launch rate, which is exactly what varies.
#
#   host arm        ttnvtop-collector normal drain -> [ring-drain] entries= / lost=
#   aggregator arm  ttnvtop-collector --fidelity-probe -> folded / lost
#
# ORDERING IS LOAD-BEARING. An aggregator can only be STARTED while a tt-metal process
# holds the device: idle_erisc.cc's wait loop is the only thing that polls
# go_messages[0], and only a tt-metal device init puts that firmware on an inactive eth
# core (5r). And when that process closes the device, the aggregator dies. So the
# workload starts FIRST, the aggregator is launched INTO it, and the workload is killed
# only after the probe window closes. A first attempt at this by hand straddled the
# workload's exit and measured a 30 s window with sweeps=0.
#
# Usage (on the target host, from TT_METAL_HOME):
#   bash tt_metal/tools/ttnvtop/scripts/fidelity_ab.sh --artifact <dir> --secs 60
#
# Env: CHIPS (default all), MESH_ROWS/MESH_COLS, SIZE, SAMPLE_HZ

set -uo pipefail

TT_METAL_HOME="${TT_METAL_HOME:-$(pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR="${COLLECTOR:-${TT_METAL_HOME}/build_Release/tools/ttnvtop-collector}"
WORKLOAD_PY="${WORKLOAD_PY:-${SCRIPT_DIR}/ab_workload.py}"

ARTIFACT=""
SECS=60
MESH_ROWS="${MESH_ROWS:-1}"
MESH_COLS="${MESH_COLS:-1}"
SIZE="${SIZE:-2048}"
SAMPLE_HZ="${SAMPLE_HZ:-300}"
CHIPS="${CHIPS:-}"
HOST_ARM=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact) ARTIFACT="$2"; shift 2 ;;
    --secs)     SECS="$2"; shift 2 ;;
    --rows)     MESH_ROWS="$2"; shift 2 ;;
    --cols)     MESH_COLS="$2"; shift 2 ;;
    --size)     SIZE="$2"; shift 2 ;;
    --chips)    CHIPS="$2"; shift 2 ;;
    --no-host-arm) HOST_ARM=0; shift ;;
    -h|--help)  sed -n '2,35p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$ARTIFACT" ]] || { echo "FATAL: --artifact <dir> is required (emit it with TTNVTOP_EMIT_ARTIFACT)" >&2; exit 2; }
[[ -f "$ARTIFACT/aggregator.desc" ]] || { echo "FATAL: no aggregator.desc in $ARTIFACT" >&2; exit 2; }
[[ -x "$COLLECTOR" ]] || { echo "FATAL: no collector at $COLLECTOR" >&2; exit 2; }

DEV_ARGS=()
if [[ -n "$CHIPS" ]]; then
  IFS=',' read -ra _c <<< "$CHIPS"
  for c in "${_c[@]}"; do DEV_ARGS+=(--device "$c"); done
fi

TS="$(date +%Y%m%d-%H%M%S)"
OUT="${TT_METAL_HOME}/runs/fidelity-${TS}"
mkdir -p "$OUT"

echo "=== fidelity: host per-core drain vs on-chip aggregator ==="
echo "host     : $(hostname)"
echo "out      : ${OUT}"
echo "workload : ${MESH_ROWS}x${MESH_COLS} mesh, ${SIZE}^2 matmul"
echo "window   : ${SECS} s, both arms concurrent"
echo "chips    : ${CHIPS:-all}"
echo

WL_PID=""; HOST_PID=""
# Teardown NEVER SIGKILLs the workload. A hard kill mid-fabric freezes an active eth
# core's 0xAABB heartbeat and costs a board reset (see WL_SECS below). SIGINT reaches
# ttnn's `finally`, which closes the mesh device; give that a long grace period, and if it
# still has not exited, say so rather than forcing it.
cleanup() {
  [[ -n "$HOST_PID" ]] && kill -INT "$HOST_PID" 2>/dev/null
  if [[ -n "$WL_PID" ]] && kill -0 "$WL_PID" 2>/dev/null; then
    kill -INT "$WL_PID" 2>/dev/null
    for _ in $(seq 1 60); do
      kill -0 "$WL_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$WL_PID" 2>/dev/null; then
      echo "WARN: workload $WL_PID has not exited after 60 s. NOT killing it -- a hard kill" >&2
      echo "      mid-fabric freezes an eth heartbeat and costs a board reset. Wait it out." >&2
    fi
  fi
  return 0
}
trap cleanup EXIT INT TERM

# The workload must outlive the whole window with margin, and then EXIT CLEANLY.
#
# It is time-bounded rather than iteration-bounded and killed, because SIGKILLing a
# tt-metal process that has initialised fabric leaves the fabric ERISC firmware stopped
# mid-loop on an active ethernet core. Its heartbeat then holds FABRIC_HEARTBEAT_SIGNATURE
# (0xAABB) with a frozen counter, and UMD throws on a valid-but-frozen signature -- so the
# NEXT device open fails outright and the board needs `tt-smi -r all`. That is exactly how
# the first re-run of this script died. Give it the window plus generous margin for
# startup, the launch, the host-arm settle and the stop step.
WL_SECS=$(( SECS + 240 ))
echo "--- starting workload (holds the device; the aggregator cannot start without it) ---"
python3 "$WORKLOAD_PY" --rows "$MESH_ROWS" --cols "$MESH_COLS" --size "$SIZE" \
    --seconds "$WL_SECS" --warmup 200 --label fidelity > "$OUT/workload.log" 2>&1 &
WL_PID=$!

for i in $(seq 1 180); do
  grep -q "Config{" "$OUT/workload.log" 2>/dev/null && break
  kill -0 "$WL_PID" 2>/dev/null || { echo "FATAL: workload died during startup; see $OUT/workload.log" >&2; exit 1; }
  sleep 1
done
sleep 20   # let it get past warmup and into steady-state kernel launches
kill -0 "$WL_PID" 2>/dev/null || { echo "FATAL: workload died before the window" >&2; exit 1; }
echo "    workload up (pid $WL_PID)"

echo "--- launching the aggregator into it ---"
"$COLLECTOR" --launch-aggregator "$ARTIFACT" "${DEV_ARGS[@]}" > "$OUT/launch.log" 2>&1
grep -E "RUNNING|NOT RUNNING|launched" "$OUT/launch.log" | sed 's/^/    /'
# "NOT RUNNING (sweeps" contains "RUNNING (sweeps", so match the em-dash prefix the
# launcher prints only on success. Getting this wrong makes the guard always pass.
grep -q -- "— RUNNING (sweeps" "$OUT/launch.log" || { echo "FATAL: no aggregator started; see $OUT/launch.log" >&2; exit 1; }
STARTED=$(grep -c -- "— RUNNING (sweeps" "$OUT/launch.log")
NOTSTARTED=$(grep -c -- "NOT RUNNING" "$OUT/launch.log")
echo "    aggregators started: ${STARTED}, failed: ${NOTSTARTED}"

if [[ $HOST_ARM -eq 1 ]]; then
  echo "--- starting the host per-core drain arm ---"
  "$COLLECTOR" "${DEV_ARGS[@]}" --sample-hz "$SAMPLE_HZ" > "$OUT/host_arm.log" 2>&1 &
  HOST_PID=$!
  sleep 8
  if ! kill -0 "$HOST_PID" 2>/dev/null; then
    echo "    WARN: host arm exited early (Wormhole-only?); see $OUT/host_arm.log"
    HOST_PID=""
    HOST_ARM=0
  fi
fi

# Mark where the window starts in the host arm's cumulative log, so the diff below is
# over the SAME interval the aggregator arm measures and not over the collector's whole
# lifetime.
HOST_MARK=0
if [[ $HOST_ARM -eq 1 ]]; then
  HOST_MARK=$(grep -c "ring-drain" "$OUT/host_arm.log" 2>/dev/null || echo 0)
fi

echo "--- ${SECS} s window ---"
"$COLLECTOR" --fidelity-probe "$SECS" "${DEV_ARGS[@]}" 2>&1 | tee "$OUT/agg_arm.log" \
  | grep -vE "UMD \||Low power|TopologyDiscovery|firmware bundle"

# Stop the aggregator BEFORE the workload closes the device.
#
# Asking it to return hands the eth core back to idle_erisc.cc, which resumes its own
# heartbeat. Letting the workload exit first kills the kernel where it stands and leaves
# our 0xABCD heartbeat word frozen with a valid signature -- which tt-metal turns into a
# hard error on the next device open and a board reset to clear. On Wormhole that costs
# real time, so it is done here, in order, and confirmed.
echo
echo "--- stopping the aggregator (while the workload still holds the device) ---"
"$COLLECTOR" --stop-aggregator "${DEV_ARGS[@]}" 2>&1 | grep -E "STOPPED|DID NOT STOP|stopped|nothing to stop" | sed 's/^/    /'

if [[ $HOST_ARM -eq 1 ]]; then
  kill -INT "$HOST_PID" 2>/dev/null; wait "$HOST_PID" 2>/dev/null; HOST_PID=""
  echo
  echo "=== HOST ARM — same window ==="
  python3 - "$OUT/host_arm.log" "$HOST_MARK" <<'PY'
import re, sys
lines = [l for l in open(sys.argv[1], errors="replace") if "[ring-drain]" in l]
mark = int(sys.argv[2])
lines = lines[mark:]
if not lines:
    print("  no [ring-drain] lines inside the window — the host arm produced no drain stats.")
    raise SystemExit(0)
def parse(l):
    d = dict(re.findall(r"(\w+)=([0-9.]+)", l))
    return int(d.get("chip", -1)), d
first, last = {}, {}
for l in lines:
    c, d = parse(l)
    first.setdefault(c, d)
    last[c] = d
print("chip     entries       lost   loss%   entries/s")
tot_e = tot_l = 0
# The log is cumulative since collector start, so every number is a delta across the
# window. Using the last line alone would fold in the pre-window warm-up.
secs = None
for c in sorted(first):
    e = int(last[c]["entries"]) - int(first[c]["entries"])
    lo = int(last[c]["lost"]) - int(first[c]["lost"])
    tot_e += e; tot_l += lo
    tk = int(last[c].get("ticks", 0)) - int(first[c].get("ticks", 0))
    hz = float(last[c].get("drain_hz", 0)) or 1.0
    secs = tk / hz if hz else None
    prod = e + lo
    print(f"{c:>4} {e:>11} {lo:>10} {100.0*lo/prod if prod else 0:>7.2f} "
          f"{e/secs if secs else 0:>11.0f}")
prod = tot_e + tot_l
print(f"TOTAL entries={tot_e} lost={tot_l} loss={100.0*tot_l/prod if prod else 0:.2f}%"
      + (f"  aggregate {tot_e/secs:.0f} entries/s" if secs else ""))
PY
fi

echo
echo "artifacts: ${OUT}"
