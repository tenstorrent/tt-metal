#!/usr/bin/env bash
# Gate for the DiffusionGemma end-to-end block-diffusion RUN stage (#47464).
# Exit codes: 0 pass, 1 advisory, 2 critical, 3 checker error.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "CHECKER ERROR: not inside a git worktree" >&2; exit 3; }
cd "$ROOT" || { echo "CHECKER ERROR: cannot cd to repo root $ROOT" >&2; exit 3; }
DG="models/experimental/diffusion_gemma"

# 1) HARD gate: the shared backbone must be untouched (F1/F2, risk R-new).
bash "$HERE/check_no_shared_gemma4_edits.sh" || exit $?

# 2) The pinned device-gated RUN regression test must exist.
if [ ! -f "$DG/tests/test_device_text_demo_run.py" ]; then
  echo "CRITICAL: missing $DG/tests/test_device_text_demo_run.py (the pinned RUN regression)" >&2
  exit 2
fi

# 3) The RUN entry + demo markers must exist.
missing=0
for f in "$DG/tt/generate.py" "$DG/demo/text_demo.py"; do
  [ -f "$f" ] || { echo "CRITICAL: missing $f" >&2; missing=1; }
done
[ "$missing" -eq 0 ] || exit 2

if ! grep -q "DG_TEXT_DEMO_SUCCESS" "$DG/demo/text_demo.py"; then
  echo "ADVISORY: DG_TEXT_DEMO_SUCCESS marker not found in text_demo.py — run-outcome grep may be unreliable" >&2
  exit 1
fi

echo "OK: RUN stage artifacts present and shared backbone unmodified"
exit 0
