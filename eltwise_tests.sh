#!/usr/bin/env bash
# Run AFTER:  source setup.sh
# Guards ensure the emulator + ASAN actually ran; results are otherwise rejected.
set -uo pipefail
LOG=elt_output.log

type emule_preflight >/dev/null 2>&1 || { echo "Run 'source setup.sh' first."; exit 1; }
emule_preflight || exit 1

: > "$LOG"
for file in tests/ttnn/unit_tests/operations/eltwise/test_*.py; do
    echo "Running $file..." | tee -a "$LOG"
    pytest "$file" -v 2>&1 | tee -a "$LOG"   # 2>&1 captures [ASAN ERROR] + abort traceback
done

emule_postflight "$LOG"
