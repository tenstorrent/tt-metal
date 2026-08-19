#!/usr/bin/env bash
# Score + ingest one hardware-graded CCL op run to the shared dashboard.
set -euo pipefail
OP="$1"; OUT="$2"
TREE=/localdev/wransom/tt-metal-eval
OPGEN="$TREE/tt_metal/third_party/tt_ops_code_gen"

cd "$TREE"; source python_env/bin/activate
export PYTHONPATH="$OPGEN:$TREE"
DURATION=$(sed -n 's/.*DURATION_S=\([0-9]*\).*/\1/p' "$OUT/run_meta.txt")

cd "$OPGEN"
python -m eval.score "$TREE/ttnn/ttnn/operations/$OP" \
    --golden-results "$OUT/golden_results.txt" \
    --duration "$DURATION" --json > "$OUT/score.json"
python -c "import json;d=json.load(open('$OUT/score.json'));print('score',d.get('total_score'),d.get('grade'))"

METAL_SHA=$(git -C "$TREE" rev-parse --short HEAD)
python -m eval.ingest \
    --prompt-name "$OP" \
    --run-number "${RUN_NUMBER:-1}" \
    --starting-branch "wransom/ccl_help_allreduce_eval" \
    --starting-commit "$METAL_SHA" \
    --created-branch "${CREATED_BRANCH:-wransom/ccl_hw_bh4_$OP}" \
    --score-json "$OUT/score.json" \
    --test-results "$OUT/test_results.json" \
    --op-name "$OP" \
    --golden-name "$OP" \
    --log-dir "$OUT" \
    --runtime-backend hw
