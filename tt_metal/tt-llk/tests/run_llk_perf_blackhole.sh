#!/usr/bin/env bash
# TEST 2 -- phase locked. Everything runs, one module at a time: all 15 cores
# work through a module together and only then move on.
#
# Selection comes from a per-module id file, not from pytest-split. The first
# attempt used --splits on each file, which splits whatever it is given, so
# "shard 1 of matmul" was a different set than the shard-1 slice that had been
# compiled and 600 variants had no ELF.
set -euo pipefail
GROUP="${1:?}"; N_GROUPS="${2:?}"
SPEED_OF_LIGHT="${SPEED_OF_LIGHT:-false}"
export TT_LLK_DISABLE_ASSERTS="${TT_LLK_DISABLE_ASSERTS:-1}"
case "$SPEED_OF_LIGHT" in
  true) SPEED_OF_LIGHT_ARGS=(--speed-of-light) ;;
  false) SPEED_OF_LIGHT_ARGS=() ;;
  *) echo "SPEED_OF_LIGHT must be true or false" >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR/python_tests"
mkdir -p perf_data
export PERF_KEEP_RUNS=0
PQ="-q --override-ini=log_cli=false"
M="perf and not accuracy"

echo "===== compile the whole shard once  $(date -u +%H:%M:%S)"
pytest $PQ "${SPEED_OF_LIGHT_ARGS[@]}" --compile-producer -n 10 -m "$M" \
  --timeout=60 --splits "$N_GROUPS" --group "$GROUP" . \
  --junitxml="pytest-report-blackhole-${GROUP}-compile.xml"

echo "===== collect this shard, then split the ids by module"
pytest -q --collect-only -m "$M" --splits "$N_GROUPS" --group "$GROUP" . \
  > /tmp/all.txt 2>&1 || true
grep '::' /tmp/all.txt > /tmp/ids.txt
echo "  shard items: $(wc -l < /tmp/ids.txt)"
python3 - <<'PYS'
import collections, pathlib
ids = [l.strip() for l in pathlib.Path("/tmp/ids.txt").read_text().splitlines() if l.strip()]
by = collections.OrderedDict()
for i in ids:
    by.setdefault(i.split("::")[0], []).append(i)
for m, v in by.items():
    pathlib.Path(f"/tmp/mod_{m.replace('.py','')}.txt").write_text("\n".join(v) + "\n")
print("  modules in this shard:", {m: len(v) for m, v in by.items()})
PYS

for of in /tmp/mod_*.txt; do
  name=$(basename "$of" .txt)
  tag="${name}_${GROUP}"
  echo "===== $name  ($(wc -l < "$of") items)  $(date -u +%H:%M:%S)"
  rm -f /tmp/corelog.*
  set +e
  PERF_RUN_TAG="$tag" PERF_CORE_LOG=/tmp/corelog PERF_ORDER_FILE="$of" \
  pytest $PQ "${SPEED_OF_LIGHT_ARGS[@]}" --compile-consumer -n 15 -m "$M" \
    --timeout=60 . > "/tmp/$tag.log" 2>&1
  rc=$?
  set -e
  echo "  rc=$rc  $(tail -1 "/tmp/$tag.log")"
  D="$LLK_ROOT/perf_data/runs/corelog-$tag"; mkdir -p "$D"
  cat /tmp/corelog.*.tsv > "$D/cores.tsv" 2>/dev/null || true
done

cp "pytest-report-blackhole-${GROUP}-compile.xml" \
   "pytest-report-blackhole-${GROUP}-run.xml"
echo "===== runs:"; ls -1 "$LLK_ROOT/perf_data/runs/" | head -40
