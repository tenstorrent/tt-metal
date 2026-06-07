#!/usr/bin/env bash
# Run bilingual_s2tt.py inside the seamless container.
#
# All configuration is via environment variables (all optional):
#   REPO                 tt-metal repo root. Default: auto-detected from this script's location.
#   IMAGE                docker image. Default: tt-metalium-dev:seamless_m4t_v2_yt
#                        (base + yt-dlp + ffmpeg, needed for --youtube; superset of the base image).
#   TT_VISIBLE_DEVICES   device(s) to expose. Default: 0. For 1x4 TP=4 use "0,1,2,3".
#   SEAMLESS_FORCE_1x1   1 = force single-chip MeshShape(1,1) (default). Set empty for auto/1x4.
#   DATA_DIR             host directory to mount read-only at /data (then pass --audio /data/<file>).
#                        Default: not mounted.
#   HUGEPAGES_DIR        hugepages mount. Default: /dev/hugepages-1G (skipped if it doesn't exist).
#   SEAMLESS_M4T_V2_WEIGHTS  weights path *inside the container*. Default: the in-repo weights dir.
#
# Examples:
#   ./run_bili.sh --youtube "https://youtu.be/ID" --source-lang eng --out outputs/o.txt
#   DATA_DIR="$HOME/dat" ./run_bili.sh --audio /data/talk.wav --out outputs/o.txt
#   SEAMLESS_FORCE_1x1= TT_VISIBLE_DEVICES=0,1,2,3 ./run_bili.sh --audio /data/talk.wav   # 1x4 TP=4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_REL="models/experimental/seamless_m4t_v2_large/demo"
MODEL_REL="models/experimental/seamless_m4t_v2_large"

# This script lives at <repo>/models/experimental/seamless_m4t_v2_large/demo/run_bili.sh,
# so the repo root is four directories up. Allow REPO= to override (e.g. running a copy
# from outside the repo tree).
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
if [[ ! -f "${REPO}/${DEMO_REL}/bilingual_s2tt.py" ]]; then
  echo "ERROR: '${REPO}' does not look like a tt-metal checkout" >&2
  echo "       (missing ${DEMO_REL}/bilingual_s2tt.py). Set REPO=/path/to/tt-metal." >&2
  exit 1
fi

IMAGE="${IMAGE:-tt-metalium-dev:seamless_m4t_v2_yt}"
HUGEPAGES_DIR="${HUGEPAGES_DIR:-/dev/hugepages-1G}"
WEIGHTS="${SEAMLESS_M4T_V2_WEIGHTS:-/tt-metal/${MODEL_REL}/weights/seamless-m4t-v2-large}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "ERROR: docker image '${IMAGE}' not found. Build/commit it, or set IMAGE=<existing tag>." >&2
  exit 1
fi

docker_args=(
  --rm
  --device /dev/tenstorrent
  -v "${REPO}:/tt-metal" -w /tt-metal
  --cap-add SYS_NICE --shm-size=2g
  -e ARCH_NAME=blackhole -e TT_METAL_HOME=/tt-metal -e PYTHONPATH=/tt-metal
  -e PYTHONUNBUFFERED=1
  -e SEAMLESS_FORCE_1x1="${SEAMLESS_FORCE_1x1-1}"
  -e TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES:-0}"
  -e SEAMLESS_M4T_V2_WEIGHTS="${WEIGHTS}"
)
# Optional mounts / TTY (only when present, so the script also works in pipes / CI).
[[ -d "${HUGEPAGES_DIR}" ]] && docker_args+=(-v "${HUGEPAGES_DIR}:${HUGEPAGES_DIR}")
[[ -n "${DATA_DIR:-}" ]] && docker_args+=(-v "${DATA_DIR}:/data:ro")
[[ -t 0 && -t 1 ]] && docker_args+=(-it)

exec docker run "${docker_args[@]}" \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc 'source /tt-metal/python_env/bin/activate && exec python '"${DEMO_REL}"'/bilingual_s2tt.py "$@"' _ "$@"
