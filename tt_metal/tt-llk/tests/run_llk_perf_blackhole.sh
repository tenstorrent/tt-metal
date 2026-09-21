#!/usr/bin/env bash
# THERMAL / CLOCK TELEMETRY -- does the card throttle, and does it matter?
#
# Three arms over the same shard: the chunk that fires (374), the chunk that
# does not (10), and one worker. tt-smi is sampled throughout each.
set -euo pipefail
GROUP="${1:?}"
N_GROUPS="${2:?}"
if [ "$GROUP" != "1" ]; then
  echo "experiment: only group 1 runs; this group exits."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR/python_tests"
export PERF_KEEP_RUNS=0
unset PERF_RUN_TAG

M="perf and not accuracy"
PQ="-q --override-ini=log_cli=false"
SEL=(--splits 5 --group 1 .)
TEL=/tmp/tel
mkdir -p "$TEL"

echo "===== tt-smi availability"
tt-smi --version 2>&1 | head -2 || echo "  (no --version)"
tt-smi -s -f "$TEL/probe.json" >/dev/null 2>&1 || echo "  (snapshot -f failed)"
if [ -s "$TEL/probe.json" ]; then
  echo "----- first snapshot, first 60 lines"
  head -60 "$TEL/probe.json"
else
  echo "  no snapshot produced; telemetry will be empty"
fi

sampler() {
  # Every two seconds, one snapshot named by arm and epoch millisecond.
  while true; do
    tt-smi -s -f "$TEL/${ARM_LABEL}_$(date +%s%3N).json" >/dev/null 2>&1 || true
    sleep 2
  done
}

summarise() {
  python3 - "$TEL" "$1" <<'PYS'
import glob, json, re, sys
tel, label = sys.argv[1], sys.argv[2]
WANT = re.compile(r"aiclk|clk|clock|temp|power|voltage|current|throttl", re.I)
series = {}
for f in sorted(glob.glob(f"{tel}/{label}_*.json")):
    try:
        with open(f) as fh:
            doc = json.load(fh)
    except Exception:
        continue
    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        else:
            if WANT.search(path):
                try:
                    series.setdefault(path, []).append(float(str(node).split()[0]))
                except (ValueError, IndexError):
                    pass
    walk(doc)
print(f"  samples parsed: {len(glob.glob(f'{tel}/{label}_*.json'))}")
for k in sorted(series):
    v = series[k]
    if len(v) < 2 or min(v) == max(v) == 0:
        continue
    print(f"  {k:<52} n={len(v):<4} min={min(v):>9.1f} max={max(v):>9.1f} "
          f"mean={sum(v)/len(v):>9.1f}")
PYS
}

arm() {
  local label="$1" workers="$2" chunk="$3"
  echo "===== ARM $label  workers=$workers chunk=$chunk  $(date -u +%H:%M:%S)"
  export ARM_LABEL="$label"
  sampler & local sp=$!
  export PERF_RUN_TAG="$label"
  pytest $PQ --compile-consumer -n "$workers" -m "$M" --timeout=60 \
    --maxschedchunk "$chunk" "${SEL[@]}" > "/tmp/$label.log" 2>&1 \
    || echo "  (consumer rc=$?)"
  kill "$sp" 2>/dev/null || true
  wait "$sp" 2>/dev/null || true
  unset PERF_RUN_TAG
  tail -2 "/tmp/$label.log" | sed 's/^/  /'
  echo "----- telemetry during $label"
  summarise "$label"
}

echo "===== compiling shard 1 once  $(date -u +%H:%M:%S)"
PERF_RUN_TAG=compile pytest $PQ --compile-producer -n 10 -m "$M" --timeout=60 \
  "${SEL[@]}" > /tmp/compile.log 2>&1 || echo "  (producer rc=$?)"
tail -2 /tmp/compile.log | sed 's/^/  /'

arm t_c374   15 374
arm t_c10    15  10
arm t_n1      1 374
arm t_c374_b 15 374

echo "===== arms written:"
ls -1 "$LLK_ROOT/perf_data/runs/" || true
echo "===== experiment done ====="
