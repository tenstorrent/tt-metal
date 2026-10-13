#!/bin/bash
# t479 staging driver (no device): copy MiniMax-H3 weights to blx01 local disk, then build the tree.
set -eo pipefail
T=/var/tmp/fasth3/t479
trap 'echo $? > $T/stage.rc' EXIT
echo "[t479] copy $(date -u)"; bash $T/cp479.sh; echo "[t479] copy rc=$?"
echo "[t479] build $(date -u)"; bash $T/build479.sh; echo "[t479] build rc=$?"
echo "[t479] stage done $(date -u)"
