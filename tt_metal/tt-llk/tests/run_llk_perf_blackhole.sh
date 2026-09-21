#!/usr/bin/env bash
# SETTLING -- eight identical measuring passes, nothing else varied.
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

echo "===== tt-smi: what does it actually offer"
tt-smi --version 2>&1 | head -2 || true
echo "----- help"
tt-smi --help 2>&1 | head -50 || true
echo "----- snapshot spellings"
for form in "-s" "--snapshot" "-s -f $TEL/a.json" "--snapshot --filename $TEL/b.json"; do
  echo "  trying: tt-smi $form"
  # shellcheck disable=SC2086
  tt-smi $form >"$TEL/try.out" 2>&1 && echo "    rc=0" || echo "    rc=$?"
  head -3 "$TEL/try.out" | sed 's/^/      /'
done
echo "----- anything written"
ls -la "$TEL" 2>&1 | head -10
find . -maxdepth 2 -name "*snapshot*" -newermt "-5 minutes" 2>/dev/null | head -5 || true

sampler() {
  while true; do
    tt-smi -s -f "$TEL/${ARM_LABEL}_$(date +%s%3N).json" >/dev/null 2>&1 || \
      tt-smi -s >/dev/null 2>&1 || true
    sleep 2
  done
}

echo "===== compiling shard 1 once  $(date -u +%H:%M:%S)"
PERF_RUN_TAG=compile pytest $PQ --compile-producer -n 10 -m "$M" --timeout=60 \
  "${SEL[@]}" > /tmp/compile.log 2>&1 || echo "  (producer rc=$?)"
tail -2 /tmp/compile.log | sed 's/^/  /'

for i in 1 2 3 4 5 6 7 8; do
  label="p$i"
  echo "===== PASS $i  $(date -u +%H:%M:%S)"
  export ARM_LABEL="$label" PERF_RUN_TAG="$label"
  sampler & sp=$!
  pytest $PQ --compile-consumer -n 15 -m "$M" --timeout=60 \
    --maxschedchunk 374 "${SEL[@]}" > "/tmp/$label.log" 2>&1 \
    || echo "  (consumer rc=$?)"
  kill "$sp" 2>/dev/null || true
  wait "$sp" 2>/dev/null || true
  unset PERF_RUN_TAG ARM_LABEL
  tail -2 "/tmp/$label.log" | sed 's/^/  /'
  echo "  telemetry files: $(ls -1 "$TEL/${label}"_*.json 2>/dev/null | wc -l)"
done

echo "===== arms written:"
ls -1 "$LLK_ROOT/perf_data/runs/" || true
echo "===== experiment done ====="
