#!/usr/bin/env bash
# Runner-side gate for the diffusion datatype-sweep / readiness stage: a selected precision policy must
# be recorded and the capability contract must still be present and sane. Exit 0 pass, 1 advisory,
# 2 critical, 3 error.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../../.." && pwd)"
scripts="$repo_root/.agents/scripts"

if [ -z "${MODEL_DIR:-}" ] && [ -z "${HF_MODEL:-}" ]; then
  echo "Set MODEL_DIR or HF_MODEL to scope the check to the target model." >&2
  exit 3
fi

# selected_precision_config.json must exist somewhere under the model's doc tree.
search_root="${MODEL_DIR:-.}"
cfg="$(find "$search_root" -name selected_precision_config.json -print 2>/dev/null | head -1)"
if [ -z "$cfg" ]; then
  echo "selected_precision_config.json not found under $search_root — datatype sweep incomplete." >&2
  exit 2
fi
python - "$cfg" <<'PY' || exit 2
import json, sys
try:
    c = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"selected_precision_config.json invalid JSON: {e}", file=sys.stderr); sys.exit(2)
# must record a decision and at least one measured quality metric
if not c.get("decision") and not c.get("policy_source"):
    print("precision config records no 'decision'/'policy_source'.", file=sys.stderr); sys.exit(2)
mr = c.get("measured_result") or c.get("measured") or {}
if not any(k for k in mr if "pcc" in k.lower()):
    print("precision config records no measured PCC metric.", file=sys.stderr); sys.exit(2)
print(f"[precision] {sys.argv[1]}: decision recorded, measured PCC present.")
PY

python "$scripts/check_diffusion_contract.py" \
  --model-dir "${MODEL_DIR:-}" --hf-model "${HF_MODEL:-}" --stage datatype-sweep --require-contract
exit $?
