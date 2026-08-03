#!/usr/bin/env bash
# Runner-side gate for the TTI release stage: the customer-facing release report
# and run notes must be copied into the target model's doc/tti_release directory.
# Exit 0 pass, 1 advisory, 2 critical, 3 error.
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

release_dir="$model_dir/doc/tti_release"
if [ ! -d "$release_dir" ]; then
  echo "Missing TTI release evidence directory: $release_dir" >&2
  exit 2
fi

report_count=$(find "$release_dir" -type f -path '*/release/report_*.md' 2>/dev/null | wc -l | tr -d ' ')
if [ "$report_count" -lt 1 ]; then
  report_count=$(find "$release_dir" -maxdepth 2 -type f -name 'report_*.md' 2>/dev/null | wc -l | tr -d ' ')
fi
if [ "$report_count" -lt 1 ]; then
  echo "No release report_*.md found under $release_dir." >&2
  exit 2
fi

notes="$release_dir/RUN_NOTES.md"
if [ ! -s "$notes" ]; then
  echo "Missing non-empty RUN_NOTES.md under $release_dir." >&2
  exit 2
fi

if ! grep -Eq 'EXIT_CODE=0|exit code:?[[:space:]]*0|exited[[:space:]]+0|completed successfully' "$notes"; then
  echo "RUN_NOTES.md does not record a successful release workflow exit." >&2
  exit 2
fi

if ! grep -Eiq 'physical host|loudbox|wh-lb-|docker image|tt-inference-server|repo tag|git head|git sha' "$notes"; then
  echo "RUN_NOTES.md is missing host/version evidence expected for handoff." >&2
  exit 1
fi

if ! grep -Eiq 'autoport implementation check.*models/autoports|models/autoports/.+.*autoport implementation check' "$notes"; then
  echo "RUN_NOTES.md does not record the required autoport implementation check." >&2
  exit 2
fi

autoport_check_output=$(python3 - "$release_dir" "$model_dir" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

release_dir = Path(sys.argv[1])
target_model_dir = sys.argv[2].replace("\\", "/").strip().strip("/")


def normalize_path(value):
    text = str(value).replace("\\", "/").strip().strip("'\"")
    for marker in ("models/autoports/", "models/tt_transformers", "models/demos/"):
        index = text.find(marker)
        if index >= 0:
            return text[index:].rstrip("/")
    return text.rstrip("/")


target = normalize_path(target_model_dir)
if not target.startswith("models/autoports/"):
    raise SystemExit(f"Target model path is not an autoport path: {target_model_dir}")

json_paths = []
json_paths.extend(sorted((release_dir / "run_specs").glob("*.json")))
json_paths.extend(sorted((release_dir / "reports_output" / "release" / "data").glob("*.json")))
json_paths.extend(sorted(release_dir.glob("*model*spec*.json")))
json_paths.extend(sorted(release_dir.glob("*report*data*.json")))

if not json_paths:
    raise SystemExit(
        "No copied TTI run spec or release report data JSON found; cannot prove "
        "which implementation was evaluated."
    )


def walk(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            yield from walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, f"{path}[{index}]")
    else:
        yield path, value


autoport_hits = []
stock_hits = []
wrong_autoport_hits = []
for json_path in json_paths:
    try:
        data = json.loads(json_path.read_text())
    except Exception as exc:
        raise SystemExit(f"Could not parse {json_path}: {exc}")

    for path, value in walk(data):
        key = path.rsplit(".", 1)[-1].split("[", 1)[0]
        text = str(value)
        normalized = normalize_path(text)
        where = f"{json_path.relative_to(release_dir)}:{path}"

        if key == "code_path":
            if normalized == target:
                autoport_hits.append(where)
            elif normalized.startswith("models/autoports/"):
                wrong_autoport_hits.append(f"{where}={normalized}")
            elif normalized in {"models/tt_transformers", "models/demos/tt_transformers"} or normalized.startswith("models/demos/"):
                stock_hits.append(f"{where}={normalized}")

        if key in {"model_impl", "impl_name", "impl_id"} and text.strip().lower() in {
            "tt-transformers",
            "tt_transformers",
        }:
            stock_hits.append(f"{where}={text.strip()}")

        if key == "code_link" and (
            "models/tt_transformers" in text or "models/demos/" in text
        ):
            stock_hits.append(f"{where}={text.strip()}")

if wrong_autoport_hits:
    raise SystemExit(
        "Copied TTI artifacts point at a different autoport implementation, not "
        f"{target}: " + "; ".join(wrong_autoport_hits[:6])
    )

if stock_hits:
    raise SystemExit(
        "Copied TTI artifacts identify a stock implementation. This stage must "
        "evaluate the generated autoport model only: " + "; ".join(stock_hits[:8])
    )

if not autoport_hits:
    raise SystemExit(
        "Copied TTI artifacts do not prove that the evaluated implementation path "
        f"was {target}."
    )

print(
    "Autoport implementation check passed: "
    f"{target} found in {len(autoport_hits)} copied TTI artifact field(s)."
)
PY
)
autoport_check_status=$?
if [ "$autoport_check_status" -ne 0 ]; then
  printf '%s\n' "$autoport_check_output" >&2
  exit 2
fi
printf '%s\n' "$autoport_check_output"

python .agents/scripts/check_context_contract.py \
  --model-dir "$model_dir" --hf-model "${HF_MODEL:-}" \
  --stage tti-release --require-contract

echo "TTI release evidence present under $release_dir ($report_count release report(s))."
