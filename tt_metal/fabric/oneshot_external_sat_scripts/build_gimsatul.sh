#!/usr/bin/env bash
# Build gimsatul (the external one-shot clause-sharing SAT solver used by TT_TOPO_SAT_GIMSATUL=1) into a known dir.
# Source of truth: a local mirror if present (offline-safe), else clone from upstream.
# Result: prints the built binary path (use it as TT_TOPO_SAT_GIMSATUL_BIN).
set -eu
DEST="${1:-$HOME/gimsatul_build}"
SRC_MIRROR=/data/rsong/gimsatul_backup/gimsatul_src
mkdir -p "$DEST"
if [ -d "$SRC_MIRROR" ]; then
  cp -r "$SRC_MIRROR"/. "$DEST/src"
else
  git clone --depth 1 https://github.com/arminbiere/gimsatul "$DEST/src"
fi
cd "$DEST/src"
make clean >/dev/null 2>&1 || true
./configure >/dev/null
make -j"$(nproc)" >/dev/null
./gimsatul --version
echo "GIMSATUL_BIN=$DEST/src/gimsatul"
