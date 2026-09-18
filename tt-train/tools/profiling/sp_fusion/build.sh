#!/usr/bin/env bash
# Usage: build.sh [ninja targets...]   (default: ttnncpp ttnn _ttml, then install)
# Serialises all builds in build_Release behind one lock; installs ttnn libs into build_Release/lib, which is what the
# venv's ttnn/ttnn/_ttnn.so links against (ttnn-only edits are invisible without it). Log: $SPFUSE/logs/build-*.log
set -uo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/env.sh"
cd "$TT_METAL_HOME"
TARGETS=("$@"); [ ${#TARGETS[@]} -eq 0 ] && TARGETS=(ttnncpp ttnn _ttml)
LOG="$SPFUSE/logs/build-$(date +%H%M%S)-$$.log"
exec 9>"$SPFUSE/locks/build.lock"
echo "[build] waiting for build lock..."; flock 9; echo "[build] lock acquired $(date +%T)"
START=$(date +%s)
ninja -C build_Release "${TARGETS[@]}" 2>&1 | tee "$LOG" | grep -E "error|Error|FAILED|^\[[0-9]+/[0-9]+\] Linking" | tail -40
RC=${PIPESTATUS[0]}
if [ $RC -eq 0 ]; then ninja -C build_Release install >>"$LOG" 2>&1 || { echo "[build] install FAILED, see $LOG"; RC=1; }; fi
echo "[build] rc=$RC after $(( $(date +%s) - START ))s, full log: $LOG"
exit $RC
