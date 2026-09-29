#!/bin/bash
# tt-smi drop-in that re-pins AICLK after a reset.
#
# run_power_cases.py resets the board before every case via its --tt-smi executable, and a reset
# clears the ARC FORCE_AICLK override (the chip returns to firmware clocking: 800 MHz idle,
# 1350 MHz busy). Passing this script as --tt-smi keeps the whole sweep at a fixed clock without
# modifying tt-ember: it forwards every argument to the real tt-smi, then re-applies the force
# whenever the call included -r.
#
# Exits non-zero if either the reset or the re-pin fails, so the sweep aborts rather than
# silently collecting data at the wrong clock.
set -uo pipefail

AICLK_MHZ="${AICLK_MHZ:-1000}"
PY="${PYTHON:-python3}"
SETTER="${SET_AICLK:-$(dirname "$(readlink -f "$0")")/set_aiclk.py}"

tt-smi "$@"
rc=$?
if [ $rc -ne 0 ]; then
    echo "[aiclk-wrapper] tt-smi $* failed with $rc; not re-pinning" >&2
    exit $rc
fi

# Only a reset clears the override, so skip the ARC round-trip for query-only calls.
case " $* " in
    *" -r "*|*" --reset "*|*" -r") ;;
    *) exit 0 ;;
esac

# The ARC needs a moment after a reset before it will accept messages.
sleep 5
out=$("$PY" "$SETTER" "$AICLK_MHZ" --busy 2>&1)
echo "$out"
got=$(echo "$out" | sed -n 's/.*after force:.*AICLK.: \([0-9]\+\).*/\1/p' | tail -1)
if [ "$got" != "$AICLK_MHZ" ]; then
    echo "[aiclk-wrapper] FAILED to pin AICLK to ${AICLK_MHZ} MHz (read back '${got}')" >&2
    exit 1
fi
echo "[aiclk-wrapper] AICLK pinned to ${got} MHz"
exit 0
