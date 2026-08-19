#!/usr/bin/env bash
# Grade one generated CCL op's golden suites on REAL HARDWARE and emit the
# artifacts eval/ingest.py needs. Replaces eval_test_runner.sh's multidevice
# path, which unconditionally sets up craq-sim (no hardware mode exists yet).
set -uo pipefail

OP="$1"; OUT="$2"
TREE=/localdev/wransom/tt-metal-eval
OPGEN="$TREE/tt_metal/third_party/tt_ops_code_gen"
mkdir -p "$OUT"

cd "$TREE"
source python_env/bin/activate
export TT_METAL_HOME="$TREE" PYTHONPATH="$TREE:$OPGEN" TT_METAL_KERNEL_CACHE=/localdev/wransom/ttcache_eval
export CCL_HW_MESH_SHAPE="${CCL_HW_MESH_SHAPE:-1,4}"
export PYTEST_AXES_JSON="$OUT/test_axes.json" PYTEST_EXTRAS_JSON="$OUT/test_extras.json"

START=$(date +%s)
python -m pytest \
  "$OPGEN/eval/golden_tests/$OP/test_golden.py" \
  "$OPGEN/eval/golden_tests/$OP/test_translated.py" \
  -v --timeout=900 -p no:randomly \
  --junitxml="$OUT/junit.xml" ${EXTRA_PYTEST_ARGS:-} > "$OUT/pytest.log" 2>&1
PYTEST_RC=$?
END=$(date +%s)
DURATION=$((END-START))
echo "PYTEST_RC=$PYTEST_RC DURATION_S=$DURATION" | tee "$OUT/run_meta.txt"

cd "$OPGEN"
PYTHONPATH="$OPGEN" python - "$OUT" "$DURATION" <<'PY'
import json, sys
from pathlib import Path
from eval.classify_failures import parse_junit_xml

out = Path(sys.argv[1]); duration = int(sys.argv[2])
results = parse_junit_xml(out / "junit.xml")
(out / "test_results.json").write_text(json.dumps(results, indent=2))

counts = {}
for r in results:
    counts[r["status"]] = counts.get(r["status"], 0) + 1
passed  = counts.get("passed", 0)
failed  = counts.get("failed", 0)
errors  = counts.get("error", 0)
skipped = counts.get("skipped", 0)
hangs   = counts.get("hang", 0)
total   = len(results) - skipped          # skipped cells are out-of-scope, not graded
(out / "golden_results.txt").write_text(
    f"PASSED={passed} FAILED={failed} ERRORS={errors} SKIPPED={skipped} "
    f"HANGS={hangs} TOTAL={total}\n"
)
print(f"golden: {passed}/{total} passed  (failed={failed} errors={errors} "
      f"skipped={skipped} hangs={hangs})  duration={duration}s")
print("statuses seen:", counts)
PY
