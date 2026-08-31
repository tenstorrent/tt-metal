#!/usr/bin/env bash
# Runner-side gate for the optimized-decoder stage: the pass MUST have run
# $shard-advise (OPT-015 / shard-advise SETUP.md) and captured its output into
# the autoport. Scoped to this run's model so stale artifacts from another model
# can neither pass nor fail it. Exit 0 pass, 2 critical, 3 error.
set -u

if [ -n "${MODEL_DIR:-}" ]; then
  md="$MODEL_DIR"
elif [ -n "${HF_MODEL:-}" ]; then
  echo "shard-advise gate: MODEL_DIR not set; pass --replace MODEL_DIR=models/autoports/<model_dir> so the gate is scoped to the exact autoport." >&2
  exit 3
else
  echo "shard-advise gate: neither MODEL_DIR nor HF_MODEL is set; cannot scope the check." >&2
  exit 3
fi

sa="$md/doc/optimized_decoder/shard_advise"
report="$sa/report.json"
ir="$sa/final_ir.mlir"

miss=0
for f in "$report" "$ir"; do
  if [ ! -s "$f" ]; then
    echo "shard-advise gate: missing or empty $f" >&2
    miss=1
  fi
done
if [ "$miss" -ne 0 ]; then
  echo "CRITICAL: \$shard-advise output was not captured under $sa." >&2
  echo "The optimize stage must run 'ttnn-advise capture' (OPT-015 / .agents/skills/shard-advise/SETUP.md Part B)" >&2
  echo "and save report.json + final_ir.mlir there. Running the advisor is required; keeping its config is not." >&2
  exit 2
fi

# report.json must be valid JSON
if ! python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$report" 2>/dev/null; then
  echo "shard-advise gate: $report is not valid JSON (suspect a placeholder rather than a real capture)." >&2
  exit 2
fi

# final_ir.mlir should carry an actual traced graph (a matmul op)
if ! grep -qi "matmul" "$ir" 2>/dev/null; then
  echo "shard-advise gate: $ir has no matmul op; this does not look like a real advisor capture." >&2
  exit 2
fi

echo "shard-advise gate: OK ($report and $ir present, report.json parses, final_ir has a matmul)."
exit 0
