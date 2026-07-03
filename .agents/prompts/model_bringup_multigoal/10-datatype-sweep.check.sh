#!/usr/bin/env bash
# Runner-side gate for the post-release datatype sweep. It verifies that the
# stage produced candidate results, selected config, Pareto plot, full-rerun
# summary, and preserved the context contract. Exit 0 pass, 1 advisory, 2
# critical, 3 error.
set -uo pipefail

if [ -n "${MODEL_DIR:-}" ]; then
  model_dir="$MODEL_DIR"
elif [ -n "${HF_MODEL:-}" ]; then
  slug=$(printf '%s' "$HF_MODEL" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_//; s/_$//')
  model_dir="models/autoports/$slug"
else
  echo "Neither MODEL_DIR nor HF_MODEL is set; cannot scope the check to the target model." >&2
  exit 3
fi

sweep_dir="$model_dir/doc/datatype_sweep"
handoff="$model_dir/doc/tti_release/post_release_sweep_benchmark.json"

if [ ! -s "$handoff" ]; then
  echo "Missing TTI handoff for datatype sweep: $handoff" >&2
  exit 2
fi

for path in \
  "$sweep_dir/README.md" \
  "$sweep_dir/work_log.md" \
  "$sweep_dir/sweep_results.json" \
  "$sweep_dir/sweep_results.csv" \
  "$sweep_dir/selected_precision_config.json" \
  "$sweep_dir/benchmark_perf_pareto.png"
do
  if [ ! -s "$path" ]; then
    echo "Missing non-empty datatype-sweep artifact: $path" >&2
    exit 2
  fi
done

python3 - "$sweep_dir" "$handoff" <<'PY' || exit 2
import csv
import json
import sys
from pathlib import Path

sweep_dir = Path(sys.argv[1])
handoff_path = Path(sys.argv[2])

handoff = json.loads(handoff_path.read_text())
required_handoff = {
    "screening_benchmark",
    "screening_metric",
    "baseline_score",
    "pass_threshold",
    "performance_metric",
    "full_rerun",
}
missing_handoff = [key for key in required_handoff if key not in handoff or handoff[key] in (None, "", [], {})]
if missing_handoff:
    raise SystemExit(f"{handoff_path} missing required field(s): {', '.join(missing_handoff)}")

results = json.loads((sweep_dir / "sweep_results.json").read_text())
if isinstance(results, dict):
    candidates = results.get("candidates") or results.get("results") or results.get("evaluated_configs")
    selected_id = results.get("selected_config_id")
    full_rerun = results.get("full_rerun") or results.get("selected_full_rerun")
else:
    candidates = results
    selected_id = None
    full_rerun = None
if not isinstance(candidates, list) or not candidates:
    raise SystemExit("sweep_results.json must contain a non-empty candidate list.")

selected = json.loads((sweep_dir / "selected_precision_config.json").read_text())
if not isinstance(selected, dict):
    raise SystemExit("selected_precision_config.json must contain a JSON object.")
selected_id = selected.get("config_id") or selected.get("id") or selected_id
if not selected_id:
    raise SystemExit("selected_precision_config.json must record config_id or id.")

by_id = {str(item.get("config_id") or item.get("id")): item for item in candidates if isinstance(item, dict)}
if str(selected_id) not in by_id:
    raise SystemExit(f"selected config {selected_id!r} is not present in sweep_results.json.")
selected_result = by_id[str(selected_id)]

pass_value = selected_result.get("pass")
if pass_value is None:
    pass_value = selected_result.get("passes_screening")
if pass_value is None:
    pass_value = selected_result.get("screening_pass")
if pass_value is not True:
    raise SystemExit(f"selected config {selected_id!r} is not marked as screening pass.")

for key in ("dtype_policy", "compute_fidelity_policy"):
    if key not in selected_result and key not in selected:
        raise SystemExit(f"selected config is missing {key}.")

if not (
    selected_result.get("screening_metric")
    or selected_result.get("benchmark_metric")
    or selected_result.get("quality_metric")
):
    raise SystemExit("selected candidate is missing screening/benchmark quality metric.")
if not (selected_result.get("performance_metric") or selected_result.get("decode_tps") or selected_result.get("tps")):
    raise SystemExit("selected candidate is missing a performance metric.")

if not full_rerun:
    full_rerun = selected_result.get("full_rerun") or selected.get("full_rerun")
if not full_rerun:
    raise SystemExit("sweep_results.json or selected config must record full_rerun results for the winner.")

with (sweep_dir / "sweep_results.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("sweep_results.csv has no data rows.")
if str(selected_id) not in {row.get("config_id") or row.get("id") for row in rows}:
    raise SystemExit(f"selected config {selected_id!r} is not present in sweep_results.csv.")

print(f"Datatype sweep artifacts OK for selected config {selected_id}.")
PY

python .agents/scripts/check_context_contract.py \
  --model-dir "$model_dir" --hf-model "${HF_MODEL:-}" \
  --stage datatype-sweep --require-contract
