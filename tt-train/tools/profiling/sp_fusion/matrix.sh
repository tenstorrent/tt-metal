#!/usr/bin/env bash
# Step-time matrix through devrun: usage matrix.sh <tag> "<batches>" "<topos>" "<impls>" [MEMEFF=1 via env]
# e.g. matrix.sh c3 "1 5" "ring line" "nocomm composed fused"
set -uo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/env.sh"
TAG="${1:-$(date +%H%M)}"; BATCHES="${2:-1}"; TOPOS="${3:-ring line}"; IMPLS="${4:-nocomm composed fused}"
OUT="$SPFUSE/logs/matrix_${TAG}.txt"; : > "$OUT"; cd "$TT_METAL_HOME"
for B in $BATCHES; do for TOPO in $TOPOS; do for IMPL in $IMPLS; do
  BATCH=$B "$SPFUSE/bench_sp_train.sh" "$IMPL" "$TOPO" 6 2>&1 | grep -E "^RESULT|^PHASES|HUNG" | tee -a "$OUT"
done; done; done
echo "== done $(date +%T)" | tee -a "$OUT"
