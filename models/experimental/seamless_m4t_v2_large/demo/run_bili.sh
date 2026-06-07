#!/usr/bin/env bash
# Wrapper: run bilingual_s2tt.py in the seamless container on a single Blackhole chip.
set -euo pipefail
REPO=/home/yito/work/tt-metal
# _yt image = base + yt-dlp + ffmpeg (for --youtube). Superset of the base, safe as default.
IMAGE="${IMAGE:-tt-metalium-dev:seamless_m4t_v2_yt}"
exec docker run --rm \
  --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v "${REPO}:/tt-metal" -w /tt-metal \
  -v /home/yito/dat:/dat:ro \
  --cap-add SYS_NICE --shm-size=2g \
  -e ARCH_NAME=blackhole -e TT_METAL_HOME=/tt-metal -e PYTHONPATH=/tt-metal \
  -e PYTHONUNBUFFERED=1 -e SEAMLESS_FORCE_1x1="${SEAMLESS_FORCE_1x1-1}" \
  -e TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES:-0}" \
  -e SEAMLESS_M4T_V2_WEIGHTS=/tt-metal/models/experimental/seamless_m4t_v2_large/weights/seamless-m4t-v2-large \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc 'source /tt-metal/python_env/bin/activate && exec python models/experimental/seamless_m4t_v2_large/demo/bilingual_s2tt.py "$@"' _ "$@"
