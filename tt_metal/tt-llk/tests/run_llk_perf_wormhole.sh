#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Wormhole LLK perf runner, shared by the 5 wh matrix groups in
# tests/pipeline_reorg/llk_perf_tests.yaml (the group index is passed in).
#
# pytest-split sharding: compile this shard's items (producer), then measure
# them (consumer) -- one invocation each over the whole perf suite.
#
# Usage: SPEED_OF_LIGHT=<true|false> run_llk_perf_wormhole.sh <group> <n_groups>
set -euo pipefail

GROUP="${1:?usage: run_llk_perf_wormhole.sh <group> <n_groups>}"
N_GROUPS="${2:?usage: run_llk_perf_wormhole.sh <group> <n_groups>}"
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

# --- telemetry -------------------------------------------------------------
# Sampled for the whole script, not just the measuring pass: the compile is 80%
# of the wall clock and is where the card is coolest, so the interesting part is
# the transition into measuring.
TEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/perf_data/runs/telemetry-${GROUP}"
mkdir -p "$TEL_DIR"
TEL_CSV="$TEL_DIR/telemetry.csv"
echo "epoch_ms,phase,source,field,value" > "$TEL_CSV"
PERF_PHASE=start
export PERF_PHASE

tel_sample() {
  local ms phase
  ms=$(date +%s%3N)
  phase=$(cat /tmp/perf_phase 2>/dev/null || echo unknown)
  # tt-smi snapshot, whichever spelling works on this image.
  if tt-smi -s -f /tmp/tel_snap.json >/dev/null 2>&1 && [ -s /tmp/tel_snap.json ]; then
    python3 - "$ms" "$phase" "$TEL_CSV" <<'PYS' || true
import json, re, sys
ms, phase, out = sys.argv[1], sys.argv[2], sys.argv[3]
WANT = re.compile(r"aiclk|axiclk|arcclk|clk|clock|temp|power|voltage|current|throttl", re.I)
rows = []
def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, f"{path}[{i}]")
    elif WANT.search(path):
        try:
            rows.append((path, float(str(node).split()[0])))
        except (ValueError, IndexError):
            pass
with open("/tmp/tel_snap.json") as fh:
    walk(json.load(fh))
with open(out, "a") as fh:
    for path, val in rows:
        fh.write(f"{ms},{phase},ttsmi,{path},{val}\n")
PYS
  fi
  # Driver telemetry, if the image exposes it. Independent of tt-smi.
  for f in /sys/class/tenstorrent/*/telemetry/* /sys/class/tenstorrent/*/*clk* \
           /sys/class/tenstorrent/*/*temp*; do
    [ -r "$f" ] || continue
    v=$(head -c 64 "$f" 2>/dev/null | tr -d '\n')
    case "$v" in ''|*[!0-9.-]*) continue ;; esac
    echo "$ms,$phase,sysfs,$f,$v" >> "$TEL_CSV"
  done
}

tel_loop() { while true; do tel_sample; sleep 2; done; }

# One-time diagnostic, written beside the trace. The separate probe job for this
# outran its usefulness (a filesystem-wide find, my mistake), so the gate runs
# carry their own evidence of what telemetry this image can actually give.
{
  echo "== tt-smi --version"; tt-smi --version 2>&1 | head -3
  echo "== tt-smi --help";    tt-smi --help 2>&1 | head -40
  echo "== snapshot attempt"; tt-smi -s -f /tmp/diag.json 2>&1 | head -5
  echo "   rc=$?  size=$(stat -c%s /tmp/diag.json 2>/dev/null || echo none)"
  echo "== /sys/class/tenstorrent"; ls -la /sys/class/tenstorrent 2>&1 | head -10
  for d in /sys/class/tenstorrent/*; do
    echo "-- $d"; ls "$d" 2>&1 | head -30
  done
} > "$TEL_DIR/diagnostic.txt" 2>&1 || true

echo start > /tmp/perf_phase
tel_loop &
TEL_PID=$!
trap 'kill "$TEL_PID" 2>/dev/null || true' EXIT


PYTEST_COMPILE_EXTRA="-q --override-ini=log_cli=false"
PYTEST_RUN_EXTRA="-q --override-ini=log_cli=false"

echo compile > /tmp/perf_phase
pytest $PYTEST_COMPILE_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" --compile-producer -n 10 -m "perf and not accuracy" --timeout=60 \
  --splits "$N_GROUPS" --group "$GROUP" \
  --junitxml="pytest-report-wormhole-${GROUP}-compile.xml" .
echo measure > /tmp/perf_phase
pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" --compile-consumer --maxschedchunk 10 -n 15 -x -m "perf and not accuracy" --timeout=60 \
  --splits "$N_GROUPS" --group "$GROUP" \
  --junitxml="pytest-report-wormhole-${GROUP}-run.xml" .
junitparser merge pytest-report-wormhole-${GROUP}-compile.xml pytest-report-wormhole-${GROUP}-run.xml pytest-report-wormhole-${GROUP}.xml
