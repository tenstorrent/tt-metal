#!/usr/bin/env bash
# Critical gate for the GHCR packaging stage: exit 0 pass, exit 2 critical failure,
# exit 3 checker error.
set -u

if [[ -z "${MODEL_DIR:-}" ]]; then
    echo "MODEL_DIR is required" >&2
    exit 3
fi
if ! command -v docker >/dev/null; then
    echo "docker not available on host" >&2
    exit 3
fi

DOC_DIR="$MODEL_DIR/doc/ghcr_image"
FAIL=0

for f in README.md serve_smoke.json tt_studio_catalog_entry.json; do
    if [[ ! -s "$DOC_DIR/$f" ]]; then
        echo "CRITICAL: $DOC_DIR/$f missing or empty"
        FAIL=1
    fi
done
if [[ ! -s "$MODEL_DIR/docker/Dockerfile" ]]; then
    echo "CRITICAL: $MODEL_DIR/docker/Dockerfile missing"
    FAIL=1
fi
[[ $FAIL -ne 0 ]] && exit 2

IMAGE_REF=$(python3 - "$DOC_DIR/tt_studio_catalog_entry.json" <<'EOF'
import json, sys
entry = json.load(open(sys.argv[1]))
img = entry.get("docker_image", "")
assert img.startswith("ghcr.io/jashansinghtt/"), f"unexpected docker_image: {img!r}"
assert entry.get("service_route") == "/v1/chat/completions", "service_route must be /v1/chat/completions"
assert "P300x2" in entry.get("device_configurations", []), "device_configurations must include P300x2"
print(img)
EOF
) || { echo "CRITICAL: catalog entry invalid"; exit 2; }

echo "Catalog image ref: $IMAGE_REF"

if ! docker manifest inspect "$IMAGE_REF" >/dev/null 2>&1; then
    echo "CRITICAL: $IMAGE_REF not reachable on the registry (docker manifest inspect failed)"
    exit 2
fi

# Serve smoke must show a real chat completion.
python3 - "$DOC_DIR/serve_smoke.json" <<'EOF' || exit 2
import json, sys
blob = json.load(open(sys.argv[1]))
text = json.dumps(blob)
assert "chat/completions" in text or "choices" in text, "smoke transcript has no chat completion evidence"
assert len(text) > 300, "smoke transcript implausibly small"
EOF

# Token-leak scan on the local copy of the pushed image.
if docker pull -q "$IMAGE_REF" >/dev/null 2>&1 || docker image inspect "$IMAGE_REF" >/dev/null 2>&1; then
    if docker history --no-trunc "$IMAGE_REF" 2>/dev/null | grep -q "hf_[A-Za-z0-9]\{20,\}"; then
        echo "CRITICAL: HF token pattern found in image history"
        exit 2
    fi
    if docker image inspect "$IMAGE_REF" --format '{{json .Config.Env}}' 2>/dev/null | grep -q "hf_[A-Za-z0-9]\{20,\}"; then
        echo "CRITICAL: HF token pattern found in image env"
        exit 2
    fi
else
    echo "CRITICAL: could not pull $IMAGE_REF for token-leak scan"
    exit 2
fi

echo "ghcr-image stage: pass"
exit 0
