#!/usr/bin/env bash
# CAUSAL TEST -- force concurrent mixing up and down, hold everything else.
#
# Three arms of exactly 5,610 items over 15 workers, so the chunk is 187 in all
# three. The same 2,000 matmul tests are in every arm and only those are
# compared, so the population measured is identical and only its surroundings
# change.
set -euo pipefail
GROUP="${1:?}"; N_GROUPS="${2:?}"
if [ "$GROUP" != "1" ]; then echo "only group 1 runs"; exit 0; fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR/python_tests"
export PERF_KEEP_RUNS=0
unset PERF_RUN_TAG PERF_ORDER_FILE
M="perf and not accuracy"
PQ="-q --override-ini=log_cli=false"

echo "===== collecting the shard  $(date -u +%H:%M:%S)"
pytest -q --collect-only -m "$M" --splits 5 --group 1 . > /tmp/all.txt 2>&1 || true
grep '::' /tmp/all.txt > /tmp/ids.txt
echo "  shard items: $(wc -l < /tmp/ids.txt)"

python3 - <<'PYS'
"""Build the three order files. Same 2,000 matmul tests in all of them."""
import collections, pathlib, random

ids = [l.strip() for l in pathlib.Path("/tmp/ids.txt").read_text().splitlines() if l.strip()]
mod = lambda n: n.split("::")[0]
by = collections.OrderedDict()
for i in ids:
    by.setdefault(mod(i), []).append(i)
print("modules:", {k: len(v) for k, v in by.items()})

MATMUL = "perf_math_matmul.py"
TOTAL, SHARED = 5610, 2000
rng = random.Random(7)

mm = by.get(MATMUL, [])
assert len(mm) >= TOTAL, f"matmul has only {len(mm)} items"
shared = mm[:SHARED]                      # in every arm, same order-independent set

# arm 1: matmul only -> everything concurrent is matmul, by construction
matmul_only = shared + mm[SHARED:TOTAL]

# arm 2 and 3: the shared matmul plus other modules, to TOTAL
others = [i for m, v in by.items() if m != MATMUL for i in v]
rng.shuffle(others)
others = others[:TOTAL - SHARED]
mixed = shared + others
assert len(mixed) == TOTAL, len(mixed)

# blocked: contiguous by module -- the natural order
blocked = sorted(mixed, key=lambda n: (mod(n), n))

# interleaved: round-robin across modules, so any short window spans many
buckets = collections.OrderedDict()
for i in mixed:
    buckets.setdefault(mod(i), []).append(i)
interleaved = []
while any(buckets.values()):
    for k in list(buckets):
        if buckets[k]:
            interleaved.append(buckets[k].pop(0))
assert sorted(interleaved) == sorted(blocked), "arms differ in content"

for name, seq in (("matmul_only", matmul_only), ("blocked", blocked),
                  ("interleaved", interleaved)):
    pathlib.Path(f"/tmp/order_{name}.txt").write_text("\n".join(seq) + "\n")
    # How many distinct modules in each window of 15 consecutive items: a
    # static proxy for how mixed the workers will be.
    w = [len({mod(x) for x in seq[i:i + 15]}) for i in range(0, len(seq) - 15, 15)]
    print(f"{name:<12} n={len(seq)}  distinct modules per 15-item window: "
          f"median {sorted(w)[len(w)//2]}  mean {sum(w)/len(w):.2f}")
pathlib.Path("/tmp/order_shared.txt").write_text("\n".join(shared) + "\n")
PYS

pass_arm() {
  local label="$1" order="$2"
  echo "===== $label  $(date -u +%H:%M:%S)"
  rm -f /tmp/corelog.*
  PERF_RUN_TAG="$label" PERF_CORE_LOG=/tmp/corelog \
    PERF_ORDER_FILE="/tmp/order_${order}.txt" \
    pytest $PQ --compile-consumer -n 15 -m "$M" --timeout=60 \
    --maxschedchunk 374 . \
    > "/tmp/$label.log" 2>&1 || echo "  (rc=$?)"
  tail -2 "/tmp/$label.log" | sed 's/^/  /'
  echo "  expected 5610 items; ran $(grep -c '' /tmp/corelog.*.tsv 2>/dev/null | \
    awk -F: '{s+=$2} END {print s+0}')"
  local dest="$LLK_ROOT/perf_data/runs/corelog-$label"
  mkdir -p "$dest"
  cat /tmp/corelog.*.tsv > "$dest/cores.tsv" 2>/dev/null || true
  echo "  core log lines: $(wc -l < "$dest/cores.tsv" 2>/dev/null || echo 0)"
}

echo "===== compiling the whole shard once  $(date -u +%H:%M:%S)"
PERF_RUN_TAG=compile pytest $PQ --compile-producer -n 10 -m "$M" --timeout=60 \
  --splits 5 --group 1 . > /tmp/c.log 2>&1 || echo "  (producer rc=$?)"
tail -2 /tmp/c.log | sed 's/^/  /'

for arm in matmul_only blocked interleaved; do
  pass_arm "${arm}_a" "$arm"
  pass_arm "${arm}_b" "$arm"
done

cp /tmp/order_shared.txt "$LLK_ROOT/perf_data/runs/" 2>/dev/null || true
echo "===== runs:"; ls -1 "$LLK_ROOT/perf_data/runs/"
echo "===== done ====="
