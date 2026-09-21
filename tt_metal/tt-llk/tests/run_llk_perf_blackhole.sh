#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Blackhole LLK perf runner, shared by the 5 bh matrix groups in
# tests/pipeline_reorg/llk_perf_tests.yaml (the group index is passed in).
#
# pytest-split sharding: compile this shard's items (producer), then measure
# them (consumer) -- one invocation each over the whole perf suite.
#
# Usage: SPEED_OF_LIGHT=<true|false> run_llk_perf_blackhole.sh <group> <n_groups>
set -euo pipefail

GROUP="${1:?usage: run_llk_perf_blackhole.sh <group> <n_groups>}"
N_GROUPS="${2:?usage: run_llk_perf_blackhole.sh <group> <n_groups>}"
SPEED_OF_LIGHT="${SPEED_OF_LIGHT:-false}"
export TT_LLK_DISABLE_ASSERTS="${TT_LLK_DISABLE_ASSERTS:-1}"

case "$SPEED_OF_LIGHT" in
  true)
    SPEED_OF_LIGHT_ARGS=(--speed-of-light)
    ;;
  false)
    SPEED_OF_LIGHT_ARGS=()
    ;;
  *)
    echo "SPEED_OF_LIGHT must be 'true' or 'false', got '$SPEED_OF_LIGHT'" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python_tests"
mkdir -p perf_data

# --- experiment: warm up, then measure ------------------------------------
if [ "$GROUP" != "1" ]; then
  echo "experiment: only group 1 measures; this group exits."
  exit 0
fi
WARMUPS=5
LLK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEL_DIR="$LLK_ROOT/perf_data/runs/telemetry"
mkdir -p "$TEL_DIR"
TEL_CSV="$TEL_DIR/telemetry.csv"
echo "epoch_ms,phase,source,field,value" > "$TEL_CSV"
echo start > /tmp/perf_phase

tel_sample() {
  local ms phase
  ms=$(date +%s%3N)
  phase=$(cat /tmp/perf_phase 2>/dev/null || echo unknown)
  # -s writes to STDOUT. -f writes to ~/tt_smi/<timestamp>_snapshot.json, NOT to
  # the path given, which is what silently defeated two earlier attempts.
  if tt-smi -s --snapshot_no_tty > /tmp/snap.json 2>/dev/null && [ -s /tmp/snap.json ]; then
    python3 - "$ms" "$phase" "$TEL_CSV" <<'PYS' || true
import json, re, sys
ms, phase, out = sys.argv[1:4]
WANT = re.compile(r"clk|clock|temp|power|voltage|current|throttl|fan", re.I)
rows = []
def walk(n, p=""):
    if isinstance(n, dict):
        for k, v in n.items():
            walk(v, f"{p}.{k}" if p else k)
    elif isinstance(n, list):
        for i, v in enumerate(n):
            walk(v, f"{p}[{i}]")
    elif WANT.search(p):
        try:
            rows.append((p, float(str(n).split()[0])))
        except (ValueError, IndexError):
            pass
walk(json.load(open("/tmp/snap.json")))
with open(out, "a") as fh:
    for p, v in rows:
        fh.write(f"{ms},{phase},ttsmi,{p},{v}\n")
PYS
  fi
  for f in /sys/class/tenstorrent/*/tt_aiclk /sys/class/tenstorrent/*/tt_arcclk \
           /sys/class/tenstorrent/*/tt_axiclk \
           /sys/class/tenstorrent/*/tt_therm_trip_count; do
    [ -r "$f" ] || continue
    v=$(head -c 32 "$f" 2>/dev/null | tr -d '\n')
    case "$v" in ''|*[!0-9.-]*) continue ;; esac
    echo "$ms,$phase,sysfs,$(basename "$f"),$v" >> "$TEL_CSV"
  done
}
tel_loop() { while true; do tel_sample; sleep 3; done; }
tel_loop &
TEL_PID=$!
trap 'kill "$TEL_PID" 2>/dev/null || true' EXIT


PYTEST_COMPILE_EXTRA="-q --override-ini=log_cli=false"
PYTEST_RUN_EXTRA="-q --override-ini=log_cli=false"

pytest $PYTEST_COMPILE_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" --compile-producer -n 10 -m "perf and not accuracy" --timeout=60 \
  --splits "$N_GROUPS" --group "$GROUP" \
  --junitxml="pytest-report-blackhole-${GROUP}-compile.xml" .
# Warm-up passes, discarded. Identical to the recorded pass in every respect
# except that their output is deleted, so the only thing they contribute is
# elapsed measuring time on the card.
i=0
while [ "$i" -lt "$WARMUPS" ]; do
  i=$((i + 1))
  echo "===== WARMUP $i/$WARMUPS  $(date -u +%H:%M:%S)"
  echo "warmup$i" > /tmp/perf_phase
  PERF_RUN_TAG="warm$i" pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" \
    --compile-consumer --maxschedchunk 374 -n 15 -m "perf and not accuracy" \
    --timeout=60 --splits 5 --group 1 . > "/tmp/warm$i.log" 2>&1 \
    || echo "  (warmup rc=$?)"
  tail -2 "/tmp/warm$i.log" | sed 's/^/  /'
  rm -rf "$LLK_ROOT/perf_data/runs/warm$i"
done

echo "===== RECORDED PASS  $(date -u +%H:%M:%S)"
echo measure > /tmp/perf_phase
PERF_RUN_TAG="measure" pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" \
  --compile-consumer --maxschedchunk 374 -n 15 -x -m "perf and not accuracy" \
  --timeout=60 --splits 5 --group 1 \
  --junitxml="pytest-report-blackhole-${GROUP}-run.xml" .
echo "===== runs directory:"
ls -1 "$LLK_ROOT/perf_data/runs/"
