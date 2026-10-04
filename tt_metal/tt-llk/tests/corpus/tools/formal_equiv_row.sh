#!/usr/bin/env bash
# Run one current-tuple formal case through the canonical campaign runner.
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <row> <out_dir> [sem_node] [hand_node]" >&2
    exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS="${JO_TESTS:-$(cd "$HERE/../.." && pwd)}"
SIM="${JO_SIM:?set JO_SIM to a TTSIM_TRACE_SFPU_STREAM simulator}"

args=(
    --tests-root "$TESTS"
    --sim "$SIM"
    --out "$2"
    --ops "$1"
    --flags "${JO_FLAGS:-}"
    --timeout "${JO_TIMEOUT:-1800}"
)
if [ "$#" -eq 4 ]; then
    args+=(--sem-node "$3" --hand-node "$4")
elif [ "$#" -eq 3 ]; then
    echo "sem_node and hand_node must be supplied together" >&2
    exit 2
fi

exec "$TESTS/.venv/bin/python" "$HERE/formal_campaign.py" "${args[@]}"
