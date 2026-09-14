#!/usr/bin/env bash
# Regenerate the two single-layer Tracy captures, into a SEPARATE output dir so last night's
# captures stay intact and the two runs can be compared op-by-op.
set -u
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../pipeline_prefill_harness/wrappers
HARNESS="$(cd "$S/.." && pwd)"
cd "$(cd "$HARNESS/../../../../../.." && pwd)" || exit 1    # repo root
source "$HARNESS/env.sh"
D="${OUT_DIR:-$PWD/mistral4_singlelayer_rerun_$(date +%Y-%m-%d)}"
export M4_PROFILE_OUT="$D/profile"
mkdir -p "$M4_PROFILE_OUT"

echo "=== board health gate ($(date -Is)) ==="
if ! "$HARNESS/check_board.sh"; then
  echo "FAIL: board cannot map an 8x4 mesh -- refusing to measure (a degraded board is silently slow, not an error)."
  exit 2
fi
echo "board OK"

for mode in pp4_deep 1rank_deep; do
  echo "=== capture $mode (8 chunks) $(date -Is) ==="
  DEEP_CHUNKS=8 "$HARNESS/run_single_layer_profile.sh" "$mode"; echo "    $mode rc=$?"
done

echo "=== chips after ==="
for d in /dev/tenstorrent/[0-9]*; do p=$(fuser "$d" 2>/dev/null); [ -n "$p" ] && echo "$d: $p"; done
echo "=== END $(date -Is) ==="
