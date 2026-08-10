#!/usr/bin/env bash
# Advisory gate for the best-effort vision-tower stage: exit 0 pass, exit 1 advisory
# failure (pipeline continues to packaging either way), exit 3 checker error.
set -u

if [[ -z "${MODEL_DIR:-}" ]]; then
    echo "MODEL_DIR is required" >&2
    exit 3
fi

DOC_DIR="$MODEL_DIR/doc/vision_tower"
FAIL=0

if [[ ! -s "$DOC_DIR/README.md" ]]; then
    echo "ADVISORY: $DOC_DIR/README.md missing or empty — vision tower not delivered"
    FAIL=1
fi

SMOKE="$DOC_DIR/vllm_image_smoke.json"
if [[ ! -s "$SMOKE" ]]; then
    echo "ADVISORY: $SMOKE missing — no served image+text request recorded"
    FAIL=1
else
    python3 - "$SMOKE" <<'EOF' || FAIL=1
import json, sys
data = json.load(open(sys.argv[1]))
blob = json.dumps(data)
if len(blob) < 200:
    print("ADVISORY: image smoke transcript implausibly small")
    raise SystemExit(1)
EOF
fi

if [[ $FAIL -ne 0 ]]; then
    echo "vision-tower stage: advisory failure (text-first deliverable continues)"
    exit 1
fi
echo "vision-tower stage: pass"
exit 0
